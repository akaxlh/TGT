import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log

def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())

# negative sampling using pre-sampled entities (preSamp) for efficiency
def negSamp(temLabel, preSamp, sampSize=1000):
	negset = [None] * sampSize
	cur = 0
	for temval in preSamp:
		if temLabel[temval] == 0:
			negset[cur] = temval
			cur += 1
		if cur == sampSize:
			break
	negset = np.array(negset[:cur])
	return negset

def negSamp(temLabel, sampSize, nodeNum):
	negset = [None] * sampSize
	cur = 0
	while cur < sampSize:
		rdmItm = np.random.choice(nodeNum)
		if temLabel[rdmItm] == 0:
			negset[cur] = rdmItm
			cur += 1
	return negset


def timeProcess(trnMats):
	mi = 1e15
	ma = 0
	for i in range(len(trnMats)):
		minn = np.min(trnMats[i].data)
		maxx = np.max(trnMats[i].data)
		mi = min(mi, minn)
		ma = max(ma, maxx)
	maxTime = 0
	for i in range(len(trnMats)):
		newData = np.maximum(((trnMats[i].data - mi) / (3600*args.slot)).astype(np.int32), 1)
		# # tianchi
		# newData = trnMats[i].data - mi
		maxTime = max(np.max(newData), maxTime)
		trnMats[i] = csr_matrix((newData, trnMats[i].indices, trnMats[i].indptr), shape=trnMats[i].shape)
	print('MAX TIME', maxTime)
	return trnMats, maxTime + 1

class DataHandler:
	def __init__(self):
		if args.data == 'taobao':
			predir = './Datasets/Taobao/'
			behs = ['pv', 'fav', 'cart', 'buy']
		elif args.data == 'ijcai':
			predir = './Datasets/ijcai/'
			behs = ['click', 'fav', 'cart', 'buy']
		elif args.data == 'jd':
			predir = './Datasets/JD2021/'
			behs = ['browse', 'review', 'buy']
		self.predir = predir
		self.behs = behs
		self.trnfile = predir + 'trn_'
		self.tstfile = predir + 'tst_'

	def LoadData(self):
		trnMats = list()
		for i in range(len(self.behs)):
			beh = self.behs[i]
			path = self.trnfile + beh
			with open(path, 'rb') as fs:
				mat = pickle.load(fs)
			trnMats.append(mat)
			if args.target == 'click':
				trnLabel = (mat if i==0 else 1 * (trnLabel + mat != 0))
			elif args.target == 'buy' and i == len(self.behs) - 1:
				trnLabel = 1 * (mat != 0)
		trnMats, maxTime = timeProcess(trnMats)
		# test set
		path = self.tstfile + 'int'
		with open(path, 'rb') as fs:
			tstInt = np.array(pickle.load(fs))
		tstStat = (tstInt != None)
		tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
		self.labelP = np.squeeze(np.array(np.sum(trnLabel, axis=0)))
		self.trnMats = trnMats
		self.trnLabel = trnLabel
		self.tstInt = tstInt
		self.tstUsrs = tstUsrs
		self.maxTime = maxTime
		args.user, args.item = self.trnLabel.shape
		self.prepareGlobalData()

	def prepareGlobalData(self):
		adj = self.trnLabel.astype(np.float32)
		# adj = 0
		# for i in range(len(self.behs)):
		# 	adj = adj + self.trnMats[i]
		# adj = (adj != 0).astype(np.float32)
		tpadj = transpose(adj)
		adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
		tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
		for i in range(adj.shape[0]):
			for j in range(adj.indptr[i], adj.indptr[i+1]):
				adj.data[j] /= adjNorm[i]
		for i in range(tpadj.shape[0]):
			for j in range(tpadj.indptr[i], tpadj.indptr[i+1]):
				tpadj.data[j] /= tpadjNorm[i]
		self.adj = adj
		self.tpadj = tpadj

	def sampleLargeGraph(self, pckUsrs, pckItms=None, sampDepth=2, sampNum=args.graphSampleN, preSamp=False):
		adj = self.adj
		tpadj = self.tpadj
		def makeMask(nodes, size):
			mask = np.ones(size)
			if not nodes is None:
				mask[nodes] = 0.0
			return mask

		def updateBdgt(adj, nodes):
			if nodes is None:
				return 0
			tembat = 1000
			ret = 0
			for i in range(int(np.ceil(len(nodes) / tembat))):
				st = tembat * i
				ed = min((i+1) * tembat, len(nodes))
				temNodes = nodes[st: ed]
				ret += np.sum(adj[temNodes], axis=0)
			return ret

		def sample(budget, mask, sampNum):
			score = (mask * np.reshape(np.array(budget), [-1])) ** 2
			norm = np.sum(score)
			if norm == 0:
				return np.random.choice(len(score), 1), sampNum - 1
			score = list(score / norm)
			arrScore = np.array(score)
			posNum = np.sum(np.array(score)!=0)
			if posNum < sampNum:
				pckNodes1 = np.reshape(np.argwhere(arrScore!=0), [-1])
				# pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
				# pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
				pckNodes = pckNodes1
			else:
				pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
			return pckNodes, max(sampNum - posNum, 0)

		usrMask = makeMask(pckUsrs, adj.shape[0])
		itmMask = makeMask(pckItms, adj.shape[1])
		itmBdgt = updateBdgt(adj, pckUsrs)
		if pckItms is None:
			pckItms, _ = sample(itmBdgt, itmMask, len(pckUsrs))
			# pckItms = sample(itmBdgt, itmMask, sampNum)
			itmMask = itmMask * makeMask(pckItms, adj.shape[1])
		usrBdgt = updateBdgt(tpadj, pckItms)
		uSampRes = 0
		iSampRes = 0
		for i in range(sampDepth + 1):
			uSamp = uSampRes + (sampNum if i < sampDepth else 0)
			iSamp = iSampRes + (sampNum if i < sampDepth else 0)
			newUsrs, uSampRes = sample(usrBdgt, usrMask, uSamp)
			usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
			newItms, iSampRes = sample(itmBdgt, itmMask, iSamp)
			itmMask = itmMask * makeMask(newItms, adj.shape[1])
			if i == sampDepth or i == sampDepth and uSampRes == 0 and iSampRes == 0:
				break
			usrBdgt += updateBdgt(tpadj, newItms)
			itmBdgt += updateBdgt(adj, newUsrs)
		usrs = np.reshape(np.argwhere(usrMask==0), [-1])
		itms = np.reshape(np.argwhere(itmMask==0), [-1])
		return self.constructData(usrs, itms, pckUsrs if preSamp else None)

	def constructData(self, usrs, itms, pckUsrs=None):
		TIME, BEH, ITEM = [0, 1, 2]
		adjs = self.trnMats
		pckAdjs = []
		for i in range(len(adjs)):
			pckU = adjs[i][usrs]
			tpPckI = transpose(pckU)[itms]
			pckAdjs.append(sp.coo_matrix(transpose(tpPckI)))
		# label mat
		pckLabel = transpose(transpose(self.trnLabel[usrs])[itms])
		pckLabelP = self.labelP[itms]

		u_ut, ut_i_beh, ut_i_item, ut_i_pos = [list() for i in range(4)]
		i_ut_cols = [[list() for i in range(len(itms))] for j in range(len(self.behs))]
		user = len(usrs)
		u_i_dict = [dict() for i in range(user)]
		ut_idx = 0
		posSet = set()
		if not pckUsrs is None:
			temlen = (len(pckUsrs) * args.sampNum)
			iposLocs = dict()

		datas = [list() for i in range(user)]
		for j in range(len(self.behs)):
			row = pckAdjs[j].row
			col = pckAdjs[j].col
			data = pckAdjs[j].data
			for k in range(len(row)):
				datas[row[k]].append([data[k], j, int(col[k])])

		tst_usrDivNum, tst_usrRelNum = [list(), list()]

		for i in range(user):
			# log('A  ')
			if not pckUsrs is None and usrs[i] in pckUsrs:
				labelVec = np.reshape(pckLabel[i].toarray(), [-1])
				allLabelPos = np.reshape(np.argwhere(labelVec!=0), [-1])
				labelPos = np.random.choice(allLabelPos, args.sampNum)
				iposLocs[i] = [None] * args.sampNum
				for j in range(args.sampNum):
					iposLocs[i][j] = labelPos[j]
					posSet.add((i, labelPos[j]))

			data = datas[i]
			data.sort(key=lambda x: x[TIME])
			divnum = len(data) // args.subUsrSize + (0 if len(data) % args.subUsrSize == 0 else 1)
			tst_usrDivNum.append(divnum)
			tst_usrRelNum.append(len(data))
			if divnum == 0:
				print('USER NO ITEMS!   ')
				return [None] * 10
			for j in range(divnum):
				st, ed = [j * args.subUsrSize, (j+1) * args.subUsrSize]
				if j == divnum - 1:
					st, ed = [max(0, len(data) - args.subUsrSize), len(data)]
				# usr id starts with 0
				u_ut.append(i)
				# u_ut_time.append(data[ed-1][TIME])
				ut_i_item.append([0] * args.subUsrSize)
				ut_i_beh.append([0] * args.subUsrSize)
				ut_i_pos.append([0] * args.subUsrSize)
				for k in range(st, ed):
					# all id starts with 1
					ut_i_item[-1][k-st] = data[k][ITEM] + 1 if (i, data[k][ITEM]) not in posSet else 0
					ut_i_beh[-1][k-st] = data[k][BEH] + 1
					ut_i_pos[-1][k-st] = data[k][TIME]
					i_ut_cols[data[k][BEH]][data[k][ITEM]].append(ut_idx)
					u_i_dict[i][data[k][ITEM]] = ut_idx
				if j == divnum - 1:
					u_i_dict[i][-1] = ut_idx
				ut_idx += 1
		ut_i_item = np.array(ut_i_item)
		ut_i_beh = np.array(ut_i_beh)

		i_ut_adjs = [{'rows': [], 'cols': []} for i in range(len(self.behs))]
		for i in range(len(self.behs)):
			row = i_ut_adjs[i]['rows']
			col = i_ut_adjs[i]['cols']
			for j in range(len(itms)):
				temcols = i_ut_cols[i][j]
				for k in range(len(temcols)):
					if (u_ut[temcols[k]], j) in posSet:
						continue
					col.append(temcols[k])
					row.append(j)
		# return u_ut, u_ut_time, ut_i_beh, ut_i_item, ut_i_pos, i_ut_adjs, u_i_dict, pckLabel if pckUsrs is None else (pckLabel, iposLocs), usrs, itms
		return u_ut, ut_i_beh, ut_i_item, ut_i_pos, i_ut_adjs, u_i_dict, pckLabel if pckUsrs is None else (pckLabel, iposLocs), pckLabelP, usrs, itms