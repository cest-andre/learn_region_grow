import os
import sys
import time
import numpy

MODEL_PATH = None
BATCH_SIZE = 32
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
CURRENT_EPOCH = 0
MAX_EPOCH = 64
TRANSFORMER = False
VAL_STEP = 7
TRAIN_AREA = ['1','2','3','scannet','4','5','6']
VAL_AREA = None
#	Set to 12 as curvatures may be nan in scan data.
FEATURE_SIZE = 12
MULTISEED = 16
LITE = None
initialized = False
cross_domain = False
numpy.random.seed(0)
numpy.set_printoptions(2,linewidth=100,suppress=True,sign=' ')
for i in range(len(sys.argv)):
	if sys.argv[i]=='--train_area':
		TRAIN_AREA = sys.argv[i+1].split(',')
	if sys.argv[i]=='--val_area':
		VAL_AREA = sys.argv[i+1].split(',')
	# if sys.argv[i]=='--cross_domain':
	# 	cross_domain = True
	if sys.argv[i]=='--multiseed':
		MULTISEED = int(sys.argv[i+1])
	# if sys.argv[i]=='--lite':
	# 	LITE = int(sys.argv[i+1])

	if sys.argv[i]=='--batch_size':
		BATCH_SIZE = int(sys.argv[i+1])
	if sys.argv[i]=='--current_epoch':
		CURRENT_EPOCH = int(sys.argv[i+1])
	if sys.argv[i]=='--max_epoch':
		MAX_EPOCH = int(sys.argv[i+1])
	if sys.argv[i]=='--transformer':
		TRANSFORMER = True
	if sys.argv[i]=='--num_points':
		NUM_INLIER_POINT = NUM_NEIGHBOR_POINT = int(sys.argv[i+1])
	if sys.argv[i]=='--feature_size':
		FEATURE_SIZE = int(sys.argv[i+1])
	if sys.argv[i]=='--model_path':
		MODEL_PATH = str(sys.argv[i+1])

config = None
sess = None
saver = None

net = None

if TRANSFORMER:
	from lrg_transformer import *
	net = LrgNet_Keras(BATCH_SIZE, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE, LITE)

	net.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		run_eagerly=False
	)

	if CURRENT_EPOCH != 0:
		net.load_weights(os.path.join(MODEL_PATH, f"{CURRENT_EPOCH}ep.ckpt")).expect_partial()

else:
	from learn_region_grow_util import *

	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False

	sess = tf.compat.v1.Session(config=config)

	net = LrgNet(BATCH_SIZE, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE, LITE)

	saver = tf.compat.v1.train.Saver()

	if CURRENT_EPOCH != 0:
		saver.restore(sess, os.path.join(MODEL_PATH, f"{CURRENT_EPOCH}ep.ckpt"))
	else:
		init = tf.compat.v1.global_variables_initializer()
		sess.run(init, {})

#	Commented out as MODEL_PATH will now be passed by script argument.
#
# if cross_domain:
# 	MODEL_PATH = 'models/cross_domain/lrgnet_%s.ckpt'%TRAIN_AREA[0]
# elif FEATURE_SIZE==6:
# 	MODEL_PATH = 'models/lrgnet_model%s_xyz.ckpt'%VAL_AREA[0]
# elif FEATURE_SIZE==9:
# 	MODEL_PATH = 'models/lrgnet_model%s_xyzrgb.ckpt'%VAL_AREA[0]
# elif FEATURE_SIZE==12:
# 	MODEL_PATH = 'models/lrgnet_model%s_xyzrgbn.ckpt'%VAL_AREA[0]
# else:
# 	# use full set of features
# 	if NUM_INLIER_POINT!=512 or NUM_NEIGHBOR_POINT!=512:
# 		MODEL_PATH = 'models/lrgnet_model%s_i_%d_j_%d.ckpt'%(VAL_AREA[0], NUM_INLIER_POINT, NUM_NEIGHBOR_POINT)
# 	elif LITE is not None:
# 		MODEL_PATH = 'models/lrgnet_model%s_lite_%d.ckpt'%(VAL_AREA[0], LITE)
# 	else:
# 		MODEL_PATH = 'models/lrgnet_model%s.ckpt'%VAL_AREA[0]

epoch_time = []

for epoch in range(CURRENT_EPOCH, MAX_EPOCH):

	if not initialized or MULTISEED > 1:
		initialized = True
		train_inlier_points, train_inlier_count, train_neighbor_points, train_neighbor_count, train_add, train_remove = [], [], [], [], [], []
		val_inlier_points, val_inlier_count, val_neighbor_points, val_neighbor_count, val_add, val_remove = [], [], [], [], [], []

		if VAL_AREA is not None and epoch % VAL_STEP == VAL_STEP - 1:
			AREA_LIST = TRAIN_AREA + VAL_AREA
		else:
			AREA_LIST = TRAIN_AREA
		for AREA in AREA_LIST:
			if isinstance(AREA, str) and AREA.startswith('synthetic'):
				f = h5py.File('data/staged_%s.h5' % AREA, 'r')
			elif MULTISEED > 0 and AREA in TRAIN_AREA:
				SEED = epoch % MULTISEED
				try:
					f = h5py.File('data/multiseed/seed%d_area%s.h5'%(SEED,AREA),'r')
				except OSError:
					print("OSError")
					continue
			else:
				f = h5py.File('data/staged_area%s.h5'%(AREA),'r')
			print('Loading %s ...'%f.filename)
			if VAL_AREA is not None and AREA in VAL_AREA:
				count = f['count'][:]
				val_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					val_inlier_points.append(points[idp:idp+count[i], :FEATURE_SIZE])
					val_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:]
				val_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					val_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :FEATURE_SIZE])
					val_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
			if AREA in TRAIN_AREA:
				count = f['count'][:]
				train_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					train_inlier_points.append(points[idp:idp+count[i], :FEATURE_SIZE])
					train_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:]
				train_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					train_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :FEATURE_SIZE])
					train_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
			if FEATURE_SIZE is None:
				FEATURE_SIZE = points.shape[1]
			f.close()

		#filter out instances where the neighbor array is empty
		train_inlier_points = [train_inlier_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_inlier_count = [train_inlier_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_neighbor_points = [train_neighbor_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_add = [train_add[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_remove = [train_remove[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_neighbor_count = [train_neighbor_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		val_inlier_points = [val_inlier_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_inlier_count = [val_inlier_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_neighbor_points = [val_neighbor_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_add = [val_add[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_remove = [val_remove[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_neighbor_count = [val_neighbor_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		if len(train_inlier_points)==0:
			continue
		print('train',len(train_inlier_points),train_inlier_points[0].shape, len(train_neighbor_points))
		print('val',len(val_inlier_points), len(val_neighbor_points))

	idx = numpy.arange(len(train_inlier_points))
	numpy.random.shuffle(idx)
	inlier_points = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT, FEATURE_SIZE))
	neighbor_points = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT, FEATURE_SIZE))
	input_add = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
	input_remove = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT), dtype=numpy.int32)

	loss_arr = []
	add_prc_arr = []
	add_rcl_arr = []
	rmv_prc_arr = []
	rmv_rcl_arr = []
	num_batches = int(len(train_inlier_points) / BATCH_SIZE)
	start_time = time.time()
	for batch_id in range(num_batches):
		start_idx = batch_id * BATCH_SIZE
		end_idx = (batch_id + 1) * BATCH_SIZE
		for i in range(BATCH_SIZE):
			points_idx = idx[start_idx+i]
			N = train_inlier_count[points_idx]
			if N >= NUM_INLIER_POINT:
				subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
			else:
				subset = list(range(N)) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
			inlier_points[i,:,:] = train_inlier_points[points_idx][subset, :]
			input_remove[i,:] = train_remove[points_idx][subset]
			N = train_neighbor_count[points_idx]
			if N >= NUM_NEIGHBOR_POINT:
				subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
			else:
				subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
			neighbor_points[i,:,:] = train_neighbor_points[points_idx][subset, :]
			input_add[i,:] = train_add[points_idx][subset]

		if TRANSFORMER:
			ls = net.train_on_batch([inlier_points, neighbor_points], y=[input_remove, input_add])
		else:
			_, ls, ap, ar, rp, rr = sess.run([net.train_op, net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl],
				{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})
			
		loss_arr.append(ls)
	# 	add_prc_arr.append(ap)
	# 	add_rcl_arr.append(ar)
	# 	rmv_prc_arr.append(rp)
	# 	rmv_rcl_arr.append(rr)
	epoch_time.append(time.time() - start_time)
	# print("Epoch %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f"%(epoch+1,numpy.mean(loss_arr),numpy.mean(add_prc_arr),numpy.mean(add_rcl_arr),numpy.mean(rmv_prc_arr), numpy.mean(rmv_rcl_arr)))

	print("Epoch %d train loss %.2f"%(epoch+1,numpy.mean(loss_arr)))

	if VAL_AREA is not None and epoch % VAL_STEP == VAL_STEP - 1:
		loss_arr = []
		add_prc_arr = []
		add_rcl_arr = []
		rmv_prc_arr = []
		rmv_rcl_arr = []
		num_batches = int(len(val_inlier_points) / BATCH_SIZE)
		for batch_id in range(num_batches):
			start_idx = batch_id * BATCH_SIZE
			end_idx = (batch_id + 1) * BATCH_SIZE
			for i in range(BATCH_SIZE):
				points_idx = start_idx+i
				N = val_inlier_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
				else:
					subset = list(range(N)) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
				inlier_points[i,:,:] = val_inlier_points[points_idx][subset, :]
				input_remove[i,:] = val_remove[points_idx][subset]
				N = val_neighbor_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
				neighbor_points[i,:,:] = val_neighbor_points[points_idx][subset, :]
				input_add[i,:] = val_add[points_idx][subset]

			if TRANSFORMER:
				ls = net.test_on_batch([inlier_points, neighbor_points], y=[input_remove, input_add])
			else:
				ls, ap, ar, rp, rr = sess.run([net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl],
					{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})

			loss_arr.append(ls)
			# add_prc_arr.append(ap)
			# add_rcl_arr.append(ar)
			# rmv_prc_arr.append(rp)
			# rmv_rcl_arr.append(rr)
		# print("Validation %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f"%(epoch+1,numpy.mean(loss_arr),numpy.mean(add_prc_arr),numpy.mean(add_rcl_arr),numpy.mean(rmv_prc_arr), numpy.mean(rmv_rcl_arr)))
		print("Epoch %d validation loss %.2f"%(epoch+1,numpy.mean(loss_arr)))

	if TRANSFORMER:
		net.save_weights(os.path.join(MODEL_PATH, f"{epoch+1}ep.ckpt"))
	else:
		saver.save(sess, os.path.join(MODEL_PATH, f"{epoch+1}ep.ckpt"))

print("Avg Epoch Time: %.3f" % numpy.mean(epoch_time))
# print("GPU Mem: %.1f" % (sess.run(tf.contrib.memory_stats.MaxBytesInUse()) / 1.0e6))