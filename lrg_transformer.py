import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
from metric_loss_ops import triplet_semihard_loss

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():
	def loadFromH5(filename, load_labels=True):
		f = h5py.File(filename,'r')
		all_points = f['points'][:]
		count_room = f['count_room'][:]
		tmp_points = []
		idp = 0
		for i in range(len(count_room)):
			tmp_points.append(all_points[idp:idp+count_room[i], :])
			idp += count_room[i]
		f.close()
		room = []
		labels = []
		class_labels = []
		if load_labels:
			for i in range(len(tmp_points)):
				room.append(tmp_points[i][:,:-2])
				labels.append(tmp_points[i][:,-2].astype(int))
				class_labels.append(tmp_points[i][:,-1].astype(int))
			return room, labels, class_labels
		else:
			return tmp_points

	def savePCD(filename,points):
		if len(points)==0:
			return
		f = open(filename,"w")
		l = len(points)
		header = """# .PCD v0.7 - Point Cloud Data file format
	VERSION 0.7
	FIELDS x y z rgb
	SIZE 4 4 4 4
	TYPE F F F I
	COUNT 1 1 1 1
	WIDTH %d
	HEIGHT 1
	VIEWPOINT 0 0 0 1 0 0 0
	POINTS %d
	DATA ascii
	""" % (l,l)
		f.write(header)
		for p in points:
			rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
			f.write("%f %f %f %d\n"%(p[0],p[1],p[2],rgb))
		f.close()
		print('Saved %d points to %s' % (l,filename))

	def savePLY(filename, points):
		f = open(filename,'w')
		f.write("""ply
	format ascii 1.0
	element vertex %d
	property float x
	property float y
	property float z
	property uchar red
	property uchar green
	property uchar blue
	end_header
	""" % len(points))
		for p in points:
			f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
		f.close()
		print('Saved to %s: (%d points)'%(filename, len(points)))


	#	Returns indices of the k closest neighbors in points tensor p.
	def knn(p, k=16):
		p_1 = tf.expand_dims(p, 1)
		p_1 = tf.broadcast_to(p_1, [p_1.shape[0], p_1.shape[2], p_1.shape[2], p_1.shape[3]])

		p_2 = tf.expand_dims(p, -1)
		p_2 = tf.broadcast_to(p_2, [p_2.shape[0], p_2.shape[1], p_2.shape[2], p_2.shape[1]])
		p_2 = tf.transpose(p_2, perm=[0, 1, 3, 2])

		dist = tf.math.reduce_sum(tf.math.square(p_1 - p_2), axis=-1)

		idx = tf.argsort(dist)

		#	Exlude 0 as first neighbor will always be itself.
		return idx[:,:,1:k+1]


	#	What is share plane??  I think plane refers to feature but what is sharing?
	#	nsample is neighborhood so this will be removed to make a global attention operation.
	class PointTransformerLayer(keras.layers.Layer):
		def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
			super().__init__()
			self.mid_planes = mid_planes = out_planes // 1
			self.out_planes = out_planes
			self.share_planes = share_planes
			self.nsample = nsample
			self.linear_q = keras.layers.Dense(mid_planes)
			self.linear_k = keras.layers.Dense(mid_planes)
			self.linear_v = keras.layers.Dense(out_planes)
			self.linear_p = [keras.layers.Dense(3), keras.layers.BatchNormalization(), keras.layers.ReLU(), keras.layers.Dense(out_planes)]
			self.linear_w = [
				keras.layers.BatchNormalization(), keras.layers.ReLU(),
				keras.layers.Dense(mid_planes // share_planes),
				keras.layers.BatchNormalization(), keras.layers.ReLU(),
				keras.layers.Dense(out_planes // share_planes)
			]
			self.softmax = keras.layers.Softmax(axis=2)

		
		#	Try tf.reshape for view.
		def call(self, px, idx):
			p, x = px  # (n, 3), (n, c)
			x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)

			#	Now that nsample = n (number of points), I need to perform my own grouping for x_k and x_v.
			#	This proceeds by adding an nsample dimension and storing a copy of (n, 3+c) n times.
			#	Entire dimensionality for x_k is (b, n, n, 3+c) and x_v is (b, n, n, c) (no xyz used).
			x_k = tf.concat([p, x_k], -1)
			x_k = tf.gather(x_k, idx, axis=-2, batch_dims=1)

			x_v = tf.gather(x_v, idx, axis=-2, batch_dims=1)

			x_q = tf.gather(x_q, idx, axis=-2, batch_dims=1)

			#	For full attention
			# x_k = tf.expand_dims(x_k, 1)
			# x_k = tf.broadcast_to(x_k, [x_k.shape[0], x_k.shape[2], x_k.shape[2], x_k.shape[3]])

			
			# x_v = tf.expand_dims(x_v, 1)
			# x_v = tf.broadcast_to(x_v, [x_v.shape[0], x_v.shape[2], x_v.shape[2], x_v.shape[3]])

			p_r, x_k = x_k[:, :, :, 0:3], x_k[:, :, :, 3:]
			
			for i in range(len(self.linear_p)):
				if i == 1:
					p_r = tf.transpose(self.linear_p[i](tf.transpose(p_r, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
				else:
					p_r = self.linear_p[i](p_r)

			w = x_k - x_q + tf.math.reduce_sum(tf.reshape(p_r, [p_r.shape[0], p_r.shape[1], p_r.shape[2], self.out_planes // self.mid_planes, self.mid_planes]), axis=3)

			for i in range(len(self.linear_w)):
				if i % 3 == 0:
					w = tf.transpose(self.linear_w[i](tf.transpose(w, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
				else:
					w = self.linear_w[i](w)

			w = self.softmax(w)
			b, n, nsample, c = x_v.shape; s = self.share_planes

			x = tf.reshape(tf.math.reduce_sum(tf.reshape(x_v + p_r, [b, n, nsample, s, c // s]) * tf.expand_dims(w, 3), axis=2), [b, n, c])

			return x


	class PointTransformerBlock(keras.layers.Layer):
		expansion = 1

		def __init__(self, in_planes, planes, share_planes=8, nsample=16):
			super().__init__()
			self.linear1 = keras.layers.Dense(planes, use_bias=False)
			self.bn1 = keras.layers.BatchNormalization()
			self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
			self.bn2 = keras.layers.BatchNormalization()
			self.linear3 = keras.layers.Dense(planes * self.expansion, use_bias=False)
			self.bn3 = keras.layers.BatchNormalization()
			self.relu = keras.layers.ReLU()

			# self.noise = keras.layers.GaussianNoise(1)

		def call(self, px, idx):
			p, x = px
			#	In original source code, TransformDown occurs before this call so x will have planes channels and thus
			#	identity and x will be of the same size.  Since I'm not using transformdown/up, I could add an
			#	addition lin, bn, relu and pass before setting identity but too lazy for now.
			# identity = x
			x = self.relu(self.bn1(self.linear1(x)))
			identity = x
			x = self.relu(self.bn2(self.transformer2([p, x], idx)))
			x = self.bn3(self.linear3(x))
			x += identity
			#	Add noise here to hopefully increase generalization to factory scan data.
			#	Only active during training I believe.
			# x = self.noise(x)
			x = self.relu(x)
			return [p, x]


	#	Plan is to first encode two branches, fuse features, then decode two branches.
	#	I will try to remove down/up sampling to avoid complex feature interpolation.
	#	Therefore, is really no decode phase.  There will be two superblocks for each
	#	branch, then features are fused, followed by two more superblocks, one for each branch.
	#	The second round of blocks will have final output layers as well.  Need to match pointnet output format.
	class PointTransformerSuperBlock(keras.layers.Layer):
		def __init__(self, block, blocks, planes, c=12, k=0):
			super().__init__()
			self.c = c
			self.k = k
			#	OOM'd.  Need to downscale.
			self.in_planes = c
			# self.in_planes, planes = c, [32, 64, 128, 256, 512]
			# self.in_planes, planes = c, [16, 32, 64, 128, 256]
			share_planes = 32
			# fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
			stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

			self.neighbors = 16

			self.encoders = []

			for i in range(len(blocks)):
				self.encoders.append(
					self._make_enc(block, planes[i], blocks[i], share_planes, stride=stride[0], nsample=nsample[0])
				)

			if k != 0:
				self.cls = [keras.layers.Dense(planes[-1] // 2), keras.layers.BatchNormalization(), keras.layers.ReLU(), keras.layers.Dense(k)]

		def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
			layers = []
			self.in_planes = planes * block.expansion
			for _ in range(blocks):
				layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))

			return layers

		#	No longer need to store intermediate outputs because I am removing point pooling for now.
		def call(self, x, idx):
			p = x[:, :, :3]

			# p, x, o = pxo  # (n, 3), (n, c), (b)
			# x = p if self.c == 3 else tf.concat([p, x], 1)
			# outputs = []

			for enc in self.encoders:
				for l in enc:
					p, x = l([p, x], idx)

				# outputs.append([p, x, o])

			if self.k == 0:
				return [p, x]
			else:
				for l in self.cls:
					x = l(x)

				return x


	class LrgNet_Keras(keras.Model):
		def __init__(self, batch_size, seq_len, num_inlier_points, num_neighbor_points, feature_size, lite=0):
			super().__init__()
			#	Initialize the branches with two super blocks, pass inputs through, pool and concat features,
			#	then initialize and pass through the new super blocks and retrieve outputs for loss.

			# with strategy.scope():

			# #	Inputs
			# self.inlier_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size*seq_len, num_inlier_points, feature_size))
			# self.neighbor_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size*seq_len, num_neighbor_points, feature_size))

			# #	Labels
			# self.add_mask_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size*seq_len, num_neighbor_points))
			# self.remove_mask_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size*seq_len, num_inlier_points))

			self.num_inlier_points = num_inlier_points
			self.inlier_idx = None
			self.neighbor_idx = None

			#	Model 2B/C/D - feature concat.
			# self.superblock_1_inlier = PointTransformerSuperBlock(PointTransformerBlock, [2], [64])
			# self.superblock_1_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [2], [64])

			# self.superblock_2_inlier = PointTransformerSuperBlock(PointTransformerBlock, [1, 1, 1], [64, 128, 512])
			# self.superblock_2_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [1, 1, 1], [64, 128, 512])

			# self.superblock_3_inlier = PointTransformerSuperBlock(PointTransformerBlock, [1, 1], [256, 128], k=2)
			# self.superblock_3_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [1, 1], [256, 128], k=2)

			# Model 4 - 4x channels of 2D.
			self.superblock_1_inlier = PointTransformerSuperBlock(PointTransformerBlock, [2], [256])
			self.superblock_1_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [2], [256])

			self.superblock_2_inlier = PointTransformerSuperBlock(PointTransformerBlock, [1, 1, 1], [256, 512, 2048])
			self.superblock_2_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [1, 1, 1], [256, 512, 2048])

			self.superblock_3_inlier = PointTransformerSuperBlock(PointTransformerBlock, [1, 1], [1024, 512], k=2)
			self.superblock_3_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [1, 1], [1024, 512], k=2)

			# 	Model 5.  Original PointTransformer arch with feature
			# self.superblock_1_inlier = PointTransformerSuperBlock(PointTransformerBlock, [2, 3, 4], [32, 64, 128])
			# self.superblock_1_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [2, 3, 4], [32, 64, 128])

			# self.superblock_2_inlier = PointTransformerSuperBlock(PointTransformerBlock, [6], [256])
			# self.superblock_2_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [6], [256])

			# self.superblock_3_inlier = PointTransformerSuperBlock(PointTransformerBlock, [3], [512], k=2)
			# self.superblock_3_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [3], [512], k=2)

			# # 	Model 5A.
			# self.superblock_1_inlier = PointTransformerSuperBlock(PointTransformerBlock, [2, 3], [64, 128])
			# self.superblock_1_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [2, 3], [64, 128])

			# self.superblock_2_inlier = PointTransformerSuperBlock(PointTransformerBlock, [4, 6], [256, 512])
			# self.superblock_2_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [4, 6], [256, 512])

			# self.superblock_3_inlier = PointTransformerSuperBlock(PointTransformerBlock, [3], [1024], k=2)
			# self.superblock_3_neighbor = PointTransformerSuperBlock(PointTransformerBlock, [3], [1024], k=2)

			
		def call(self, x):
			inlier_pl, neighbor_pl = x

			self.inlier_idx = knn(inlier_pl[:, :, :3])
			self.neighbor_idx = knn(neighbor_pl[:, :, :3])

			#	Model 5 forward pass.
			# _, remove_feats = self.superblock_1_inlier(inlier_pl, self.inlier_idx)
			# _, add_feats = self.superblock_1_neighbor(neighbor_pl, self.neighbor_idx)

			# #	Pool and pass to second superblock set.
			# feature_pool = tf.concat(axis=-1, values=[
			# 		tf.reduce_max(input_tensor=remove_feats, axis=1, keepdims=True),
			# 		tf.reduce_max(input_tensor=add_feats, axis=1, keepdims=True)
			# 	]
			# )

			# feature_pool = tf.broadcast_to(feature_pool, [feature_pool.shape[0], self.num_inlier_points, feature_pool.shape[2]])

			# remove_inter = tf.concat(axis=-1, values=[remove_feats, feature_pool])
			# add_inter = tf.concat(axis=-1, values=[add_feats, feature_pool])

			# remove_output = self.superblock_2_inlier(remove_inter, self.inlier_idx)
			# add_output = self.superblock_2_neighbor(add_inter, self.neighbor_idx)


			#	Original LrgNet arch forward pass.
			_, remove_inter = self.superblock_1_inlier(inlier_pl, self.inlier_idx)
			_, add_inter = self.superblock_1_neighbor(neighbor_pl, self.neighbor_idx)

			_, remove_feats = self.superblock_2_inlier(remove_inter, self.inlier_idx)
			_, add_feats = self.superblock_2_neighbor(add_inter, self.neighbor_idx)

			#	Pool and pass to second superblock set.
			feature_pool = tf.concat(axis=-1, values=[
					tf.reduce_max(input_tensor=remove_feats, axis=1, keepdims=True),
					tf.reduce_max(input_tensor=add_feats, axis=1, keepdims=True)
				]
			)

			feature_pool = tf.broadcast_to(feature_pool, [feature_pool.shape[0], self.num_inlier_points, feature_pool.shape[2]])

			remove_inter = tf.concat(axis=-1, values=[remove_inter, feature_pool])
			add_inter = tf.concat(axis=-1, values=[add_inter, feature_pool])

			remove_output = self.superblock_3_inlier(remove_inter, self.inlier_idx)
			add_output = self.superblock_3_neighbor(add_inter, self.neighbor_idx)

			self.inlier_idx = None
			self.neighbor_idx = None

			return [remove_output, add_output]


		def train_step(self, data):
			inputs, labels = data
			remove_mask_pl, add_mask_pl = labels

			with tf.GradientTape() as tape:
				remove_output, add_output = self(inputs, training=True)

				add_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=add_output, labels=add_mask_pl))

				pos_mask = tf.where(tf.cast(remove_mask_pl, tf.bool))
				neg_mask = tf.where(tf.cast(1 - remove_mask_pl, tf.bool))
				pos_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(remove_output, pos_mask), labels=tf.gather_nd(remove_mask_pl, pos_mask)))
				neg_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(remove_output, neg_mask), labels=tf.gather_nd(remove_mask_pl, neg_mask)))
				pos_loss = tf.cond(pred=tf.math.is_nan(pos_loss), true_fn=lambda: 0.0, false_fn=lambda: pos_loss)
				neg_loss = tf.cond(pred=tf.math.is_nan(neg_loss), true_fn=lambda: 0.0, false_fn=lambda: neg_loss)

				remove_loss = pos_loss + neg_loss

				loss = add_loss + remove_loss

			trainable_vars = self.trainable_variables
			gradients = tape.gradient(loss, trainable_vars)
			self.optimizer.apply_gradients(zip(gradients, trainable_vars))

			return {"loss": loss}


		def test_step(self, data):
			inputs, labels = data
			remove_mask_pl, add_mask_pl = labels

			remove_output, add_output = self(inputs, training=True)

			add_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=add_output, labels=add_mask_pl))

			pos_mask = tf.where(tf.cast(remove_mask_pl, tf.bool))
			neg_mask = tf.where(tf.cast(1 - remove_mask_pl, tf.bool))
			pos_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(remove_output, pos_mask), labels=tf.gather_nd(remove_mask_pl, pos_mask)))
			neg_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(remove_output, neg_mask), labels=tf.gather_nd(remove_mask_pl, neg_mask)))
			pos_loss = tf.cond(pred=tf.math.is_nan(pos_loss), true_fn=lambda: 0.0, false_fn=lambda: pos_loss)
			neg_loss = tf.cond(pred=tf.math.is_nan(neg_loss), true_fn=lambda: 0.0, false_fn=lambda: neg_loss)

			remove_loss = pos_loss + neg_loss

			loss = add_loss + remove_loss

			return {"loss": loss}