import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		args.user, args.item = self.handler.trnLabel.shape
		args.behNums = len(self.handler.trnMats)
		self.maxTime = self.handler.maxTime
		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

		self.u_ut_0, self.ut_i_beh_0, self.ut_i_item_0, self.ut_i_time_0, self.i_ut_adjs_0, self.u_i_dict_0, self.pckLabel_0, self.labelP_0, _, _ = self.handler.constructData(list(range(args.user)), list(range(args.item)))

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def sequenceModeling(self, lats):
		itemEmbeds = tf.nn.embedding_lookup(lats, self.ut_i_item - 1)
		posEmbeds = tf.nn.embedding_lookup(self.timeEmbeds, self.ut_i_time)
		behEmbeds = tf.nn.embedding_lookup(self.behEmbeds, self.ut_i_beh - 1)
		posEmbeds = tf.reshape(FC(tf.reshape(posEmbeds, [-1, args.latdim * 2]), args.latdim, reg=True, useBias=True, activation=self.actFunc), [-1, args.subUsrSize, args.latdim])
		behEmbeds = tf.reshape(FC(tf.reshape(behEmbeds, [-1, args.latdim]), args.latdim, reg=True, useBias=True, activation=self.actFunc), [-1, args.subUsrSize, args.latdim])
		biasEmbed = posEmbeds + behEmbeds
		embeds = (itemEmbeds + biasEmbed) * tf.expand_dims(tf.to_float(tf.sign(self.ut_i_item)), [-1])
		Q = NNs.defineRandomNameParam([args.latdim, args.att_head, args.latdim//args.att_head], reg=True)
		K = NNs.defineRandomNameParam([args.latdim, args.att_head, args.latdim//args.att_head], reg=True)
		q = tf.expand_dims(tf.einsum('ijk,klm->ijlm', embeds, Q), axis=2)
		k = tf.expand_dims(tf.einsum('ijk,klm->ijlm', embeds, K), axis=1)
		v = tf.reshape(embeds, [-1, 1, args.subUsrSize, args.att_head, args.latdim//args.att_head])
		logits = tf.reduce_sum(q * k, axis=-1, keepdims=True)
		exp = tf.math.exp(logits) * (1.0 - tf.to_float(tf.equal(logits, 0.0)))
		norm = (tf.reduce_sum(exp, axis=2, keepdims=True) + 1e-6)
		att = exp / norm
		ret = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, args.subUsrSize, args.latdim]) + embeds
		return ret/2

	def handleMultBehEmbeds(self, embeds):
		# attentive aggregation
		glbQuery = FC(tf.add_n(embeds), args.latdim, activation=self.actFunc, useBias=True, reg=True)
		weights = []
		for embed in embeds:
			weights.append(tf.reduce_sum(embed * glbQuery, axis=-1, keepdims=True))
		stkWeight = tf.concat(weights, axis=1)
		sftWeight = tf.reshape(tf.nn.softmax(stkWeight, axis=1), [-1, args.behNums, 1]) * 16
		stkLat = tf.stack(embeds, axis=1)
		lat = tf.reshape(tf.reduce_sum(sftWeight * stkLat, axis=1), [-1, args.latdim])
		return lat

	def itemToUser(self, ilats):
		embeds = []
		paramId = 'dfltP%d' % NNs.getParamId()
		for i in range(args.behNums):
			mask = tf.to_float(tf.expand_dims(tf.equal(self.ut_i_beh - 1, i), axis=-1))
			embed = tf.reduce_sum(mask * ilats, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-6)
			behTrans_w = FC(tf.expand_dims(self.behEmbeds[i], axis=0), args.memosize, activation='relu', useBias=True, reg=True, name=paramId+'_a', reuse=True)
			behTrans = tf.reshape(FC(behTrans_w, args.latdim ** 2, reg=True, name=paramId+'_b', reuse=True), [args.latdim, args.latdim])
			embed = Activate(embed @ behTrans, self.actFunc)
			embeds.append(embed)
		return self.handleMultBehEmbeds(embeds)

	def aggregateTimeAwareULat(self, utlat):
		decayedEmbed = utlat * tf.expand_dims(self.u_ut_pos, axis=-1)
		ulat = tf.math.segment_sum(decayedEmbed, self.u_ut)
		return ulat

	# def aggregateSubUsers(self, utlat, ulat):
	# 	query = tf.nn.embedding_lookup(FC(ulat, args.latdim, reg=True), self.u_ut)
	# 	key = FC(utlat, args.latdim, reg=True)
	# 	value = tf.reshape(FC(utlat, args.latdim, reg=True), [-1, args.att_head, args.latdim//args.att_head])
	# 	# value = tf.reshape(utlat, [-1, args.att_head, args.latdim//args.att_head])
	# 	logits = tf.math.exp(tf.reduce_sum(tf.reshape(query * key, [-1, args.att_head, args.latdim//args.att_head]), axis=-1, keepdims=True) / tf.sqrt(args.latdim / args.att_head))
	# 	attNorm = tf.nn.embedding_lookup(tf.math.segment_sum(logits, self.u_ut), self.u_ut) + 1e-6
	# 	att = logits / attNorm# * 8
	# 	attval = tf.reshape(tf.math.segment_sum(att * value, self.u_ut), [-1, args.latdim])
	# 	return attval

	def aggregateSubUsers(self, utlat):
		ulat = tf.math.segment_sum(utlat, self.u_ut) / tf.expand_dims(tf.to_float(tf.math.segment_sum(tf.ones_like(self.u_ut), self.u_ut)) + 1e-6, axis=-1)
		return ulat

	def userToItem(self, utlats):
		embeds = []
		paramId = 'dfltP%d' % NNs.getParamId()
		for i in range(args.behNums):
			adj = self.i_ut_adjs[i]
			rows, cols = adj['rows'], adj['cols']
			colLats = tf.nn.embedding_lookup(utlats, cols)
			ones = tf.concat([tf.ones_like(rows), [0]], axis=0)
			rows = tf.concat([rows, [args.item-1]], axis=0)
			colLats = tf.concat([colLats, tf.zeros([1, args.latdim])], axis=0)
			embed = tf.math.segment_sum(colLats, rows) / (tf.to_float(tf.expand_dims(tf.math.segment_sum(ones, rows), axis=-1)) + 1e-6)
			behTrans_w = FC(tf.expand_dims(self.behEmbeds[i], axis=0), args.memosize, activation='relu', useBias=True, reg=True, name=paramId+'_a', reuse=True)
			behTrans = tf.reshape(FC(behTrans_w, args.latdim ** 2, reg=True, name=paramId+'_b', reuse=True), [args.latdim, args.latdim])
			embed = Activate(embed @ behTrans, self.actFunc)
			embeds.append(embed)
		return self.handleMultBehEmbeds(embeds)

	def getTimeAwareULats(self, lats):
		embeds0 = tf.nn.embedding_lookup(lats, self.u_ut)
		embeds1 = FC(tf.nn.embedding_lookup(self.timeEmbeds, self.u_ut_time), args.latdim, reg=True, useBias=True, activation=self.actFunc)
		embeds1 = FC(embeds1, args.latdim, reg=True, useBias=True, activation=self.actFunc) + embeds1
		return embeds0 + embeds1

	def makeTimeEmbed(self):
		divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
		pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
		if pos.shape[0] != self.maxTime:
			pos = tf.concat([pos, [[self.maxTime-1]]], axis=0)
		sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim*2])
		return timeEmbed

	def generateEmbeds(self):
		self.behEmbeds = NNs.defineParam('behEmbeds', [args.behNums, args.latdim], reg=False)
		self.timeEmbeds = self.makeTimeEmbed()

	def ours(self):
		self.generateEmbeds()
		uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
		iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)

		ulats = [uEmbed0]
		ilats = [iEmbed0]
		utlats = [None]
		for i in range(args.gnn_layer):
			# i to u
			ilat0 = self.sequenceModeling(ilats[-1])
			utlat = self.itemToUser(ilat0)
			ulat = self.aggregateSubUsers(utlat)
			ulats.append(ulat)

			# u to i
			utlat0 = self.getTimeAwareULats(ulats[-2])
			utlats.append(utlat0)
			ilat = self.userToItem(utlat0)
			ilats.append(ilat)
		utlats[0] = self.getTimeAwareULats(NNs.defineParam('uEmbed0Pred', [args.user, args.latdim], reg=True))
		ilats[0] = NNs.defineParam('iEmbed0Pred', [args.item, args.latdim], reg=True)
		utlat = tf.add_n(utlats)
		ulat = tf.add_n(ulats)
		ilat = tf.add_n(ilats)
		pckULat = tf.nn.embedding_lookup(utlat, self.utids)
		pckILat = tf.nn.embedding_lookup(ilat, self.iids)

		predLat = pckULat * pckILat * args.mult

		for i in range(args.deep_layer):
			predLat = FC(predLat, args.latdim, reg=True, useBias=True, activation=self.actFunc) + predLat
		pred = tf.squeeze(FC(predLat, 1, reg=True, useBias=True))
		return pred

	def prepareModel(self):
		self.actFunc = 'relu'

		self.u_ut = tf.constant(self.u_ut_0)
		self.ut_i_time = tf.constant(self.ut_i_time_0)
		self.ut_i_beh = tf.constant(self.ut_i_beh_0)
		self.ut_i_item = tf.constant(self.ut_i_item_0)
		self.i_ut_adjs = list()
		for i in range(args.behNums):
			self.i_ut_adjs.append({
				'rows': tf.constant(self.i_ut_adjs_0[i]['rows']),
				'cols': tf.constant(self.i_ut_adjs_0[i]['cols'])
			})

		self.utids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])

		self.u_ut_time = tf.reduce_max(self.ut_i_time, axis=-1)

		self.pred = self.ours()
		sampNum = tf.shape(self.iids)[0] // 2
		posPred = tf.slice(self.pred, [0], [sampNum])
		negPred = tf.slice(self.pred, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, itmNum, label, u_i_dict):
		temLabel = label[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			neglocs = negSamp(temLabel[i], sampNum, itmNum)
			poslocs = np.random.choice(posset, args.sampNum)
			neglocs = np.random.choice(negset, args.sampNum)
			for j in range(args.sampNum):
				uIntLoc[cur] = uIntLoc[cur+temlen//2] = u_i_dict[batIds[i]][poslocs[j]]
				iIntLoc[cur] = poslocs[j]
				iIntLoc[cur+temlen//2] = neglocs[j]
				cur += 1
		return uIntLoc, iIntLoc

	def sampleTrainBatch(self, batIds, itmNum, labelMat, u_i_dict):
		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(args.sampNum, len(posset))
			if sampNum == 0:
				poslocs = []
				neglocs = []
			else:
				poslocs = np.random.choice(posset, sampNum)
				neglocs = negSamp(temLabel[i], sampNum, itmNum)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = u_i_dict[batIds[i]][posloc]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		return uLocs, iLocs

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)

		feeddict = dict()

		steps = int(np.ceil(num / args.batch))
		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			uLocs, iLocs = self.sampleTrainBatch(batIds, args.item, self.pckLabel_0, self.u_i_dict_0)

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			feeddict[self.utids] = uLocs
			feeddict[self.iids] = iLocs
			res = self.sess.run(target, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batIds, label, labelP, tstInt, u_i_dict):
		batch = len(batIds)
		temTst = tstInt[batIds]
		temLabel = label[batIds].toarray()
		temlen = batch * 100
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			# rdmNegSet = np.random.permutation(negset)[:99]
			pvec = labelP[negset]
			pvec = pvec / np.sum(pvec)
			rdmNegSet = np.random.choice(negset, 99, replace=False)#, p=pvec)
			locset = np.concatenate((rdmNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uIntLoc[cur] = u_i_dict[batIds[i]][-1]
				iIntLoc[cur] = locset[j]
				cur += 1
		return uIntLoc, iIntLoc, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg, epochMrr = [0] * 3
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch#np.maximum(1, args.batch * args.sampNum * 2 // 100)

		steps = int(np.ceil(num / tstBat))

		pckTstInt = self.handler.tstInt
		feeddict = dict()

		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.pckLabel_0, self.labelP_0, pckTstInt, self.u_i_dict_0)
			feeddict[self.utids] = uLocs
			feeddict[self.iids] = iLocs
			preds = self.sess.run(self.pred, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			hit, ndcg, mrr = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			epochMrr += mrr
			log('Steps %d/%d: hit = %d, ndcg = %d, mrr = %d          ' % (i, steps, hit, ndcg, mrr), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		ret['MRR'] = epochMrr / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		mrr = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
				mrr += np.reciprocal(shoot.index(temTst[j]) + 1)
		return hit, ndcg, mrr
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()