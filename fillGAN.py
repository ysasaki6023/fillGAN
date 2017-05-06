import os,path,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time

class BatchGenerator:
    def __init__(self):
        self.folderPath = "celebA"
        self.imagePath = glob.glob(self.folderPath+"/*.jpg")
        #self.orgSize = (218,173)
        self.imgSize = (108,108)
        self.dropSize = (32,32)
        assert self.imgSize[0]==self.imgSize[1]

    def getBatch(self,nBatch):
        x1   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        x2   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        mm   = np.ones ( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        for i in range(nBatch):
            img = cv2.imread(self.imagePath[random.randint(0,len(self.imagePath)-1)])
            dmin = min(img.shape[0],img.shape[1])
            img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
            img = cv2.resize(img,self.imgSize)
            x1[i,:,:,:] = (img - 128.) / 255. # normalize between -0.5 ~ +0.5 <- requirements from using tanh in the last processing in the Generator

            pos0 = np.random.randint(int(self.dropSize[0]/2)+1,int(self.imgSize[0]-self.dropSize[0]/2)-1)
            pos1 = np.random.randint(int(self.dropSize[1]/2)+1,int(self.imgSize[1]-self.dropSize[1]/2)-1)
            py1 = int(pos0-self.dropSize[0]/2)
            py2 = int(pos0+self.dropSize[0]/2)
            px1 = int(pos1-self.dropSize[1]/2)
            px2 = int(pos1+self.dropSize[1]/2)
            img[py1:py2,px1:px2,:] = 255.

            x2[i,:,:,:] = (img - 128.) / 255. # normalize between -0.5 ~ +0.5 <- requirements from using tanh in the last processing in the Generator
            mm[i,py1:py2,px1:px2,:] = 0

        return x1,x2,mm

class fillGAN:
    def __init__(self,isTraining,imageSize,args):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = imageSize
        self.saveFolder = args.saveFolder
        self.reload = args.reload
        self.buildModel()

        return

    def _fc_variable(self, weight_shape,name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _deconv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            output_channels = int(weight_shape[2])
            input_channels  = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape    , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def _deconv2d(self, x, W, output_shape, stride=1):
        # x           : [nBatch, height, width, in_channels]
        # output_shape: [nBatch, height, width, out_channels]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x) 

    def calcImageSize(self,dh,dw,stride):
        return int(math.ceil(float(dh)/float(stride))),int(math.ceil(float(dw)/float(stride)))

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

    def buildGenerator(self,z,reuse=False,isTraining=True):
        dim_0_h,dim_0_w = self.imageSize[0],self.imageSize[1]
        dim_1_h,dim_1_w = self.calcImageSize(dim_0_h, dim_0_w, stride=2)
        dim_2_h,dim_2_w = self.calcImageSize(dim_1_h, dim_1_w, stride=2)
        dim_3_h,dim_3_w = self.calcImageSize(dim_2_h, dim_2_w, stride=2)
        dim_4_h,dim_4_w = self.calcImageSize(dim_3_h, dim_3_w, stride=2)

        h = z

        with tf.variable_scope("Generator") as scope:
            if reuse: scope.reuse_variables()

            # conv1
            self.g_conv1_w, self.g_conv1_b = self._conv_variable([5,5,3,64],name="conv1")
            h = self._conv2d(h,self.g_conv1_w, stride=2) + self.g_conv1_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm1c")
            h = self.leakyReLU(h)

            # conv2
            self.g_conv2_w, self.g_conv2_b = self._conv_variable([5,5,64,128],name="conv2")
            h = self._conv2d(h,self.g_conv2_w, stride=2) + self.g_conv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm2c")
            h = self.leakyReLU(h)

            # conv3
            self.g_conv3_w, self.g_conv3_b = self._conv_variable([5,5,128,256],name="conv3")
            h = self._conv2d(h,self.g_conv3_w, stride=2) + self.g_conv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm3c")
            h = self.leakyReLU(h)

            # conv4
            self.g_conv4_w, self.g_conv4_b = self._conv_variable([5,5,256,512],name="conv4")
            h = self._conv2d(h,self.g_conv4_w, stride=2) + self.g_conv4_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm4c")
            h = self.leakyReLU(h)

            # deconv4
            self.g_deconv4_w, self.g_deconv4_b = self._deconv_variable([5,5,512,256],name="deconv4")
            h = self._deconv2d(h,self.g_deconv4_w, output_shape=[self.nBatch,dim_3_h,dim_3_w,256], stride=2) + self.g_deconv4_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm4d")
            h = tf.nn.relu(h)

            # deconv3
            self.g_deconv3_w, self.g_deconv3_b = self._deconv_variable([5,5,256,128],name="deconv3")
            h = self._deconv2d(h,self.g_deconv3_w, output_shape=[self.nBatch,dim_2_h,dim_2_w,128], stride=2) + self.g_deconv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm3d")
            h = tf.nn.relu(h)

            # deconv2
            self.g_deconv2_w, self.g_deconv2_b = self._deconv_variable([5,5,128,64],name="deconv2")
            h = self._deconv2d(h,self.g_deconv2_w, output_shape=[self.nBatch,dim_1_h,dim_1_w,64], stride=2) + self.g_deconv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm2d")
            h = tf.nn.relu(h)

            # deconv1
            self.g_deconv1_w, self.g_deconv1_b = self._deconv_variable([5,5,64,3],name="deconv1")
            h = self._deconv2d(h,self.g_deconv1_w, output_shape=[self.nBatch,dim_0_h,dim_0_w,3], stride=2) + self.g_deconv1_b

            # sigmoid
            y = tf.tanh(h)

            ### summary
            if reuse:
                tf.summary.histogram("g_conv1_w"   ,self.g_conv1_w)
                tf.summary.histogram("g_conv1_b"   ,self.g_conv1_b)
                tf.summary.histogram("g_conv2_w"   ,self.g_conv2_w)
                tf.summary.histogram("g_conv2_b"   ,self.g_conv2_b)
                tf.summary.histogram("g_conv3_w"   ,self.g_conv3_w)
                tf.summary.histogram("g_conv3_b"   ,self.g_conv3_b)
                tf.summary.histogram("g_conv4_w"   ,self.g_conv4_w)
                tf.summary.histogram("g_conv4_b"   ,self.g_conv4_b)
                tf.summary.histogram("g_deconv1_w"   ,self.g_deconv1_w)
                tf.summary.histogram("g_deconv1_b"   ,self.g_deconv1_b)
                tf.summary.histogram("g_deconv2_w"   ,self.g_deconv2_w)
                tf.summary.histogram("g_deconv2_b"   ,self.g_deconv2_b)
                tf.summary.histogram("g_deconv3_w"   ,self.g_deconv3_w)
                tf.summary.histogram("g_deconv3_b"   ,self.g_deconv3_b)
                tf.summary.histogram("g_deconv4_w"   ,self.g_deconv4_w)
                tf.summary.histogram("g_deconv4_b"   ,self.g_deconv4_b)

        return y

    def buildDiscriminator(self,y,reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse: scope.reuse_variables()

            h = y
            # conv1
            self.d_conv1_w, self.d_conv1_b = self._conv_variable([5,5,3,64],name="conv1")
            h = self._conv2d(h,self.d_conv1_w, stride=2) + self.d_conv1_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm1")
            h = self.leakyReLU(h)

            # conv2
            self.d_conv2_w, self.d_conv2_b = self._conv_variable([5,5,64,128],name="conv2")
            h = self._conv2d(h,self.d_conv2_w, stride=2) + self.d_conv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm2")
            h = self.leakyReLU(h)

            # conv3
            self.d_conv3_w, self.d_conv3_b = self._conv_variable([5,5,128,256],name="conv3")
            h = self._conv2d(h,self.d_conv3_w, stride=2) + self.d_conv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm3")
            h = self.leakyReLU(h)

            # conv4
            self.d_conv4_w, self.d_conv4_b = self._conv_variable([5,5,256,512],name="conv4")
            h = self._conv2d(h,self.d_conv4_w, stride=2) + self.d_conv4_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm4")
            h = self.leakyReLU(h)

            # fc1
            n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
            h = tf.reshape(h,[self.nBatch,n_h*n_w*n_f])
            self.d_fc1_w, self.d_fc1_b = self._fc_variable([n_h*n_w*n_f,1],name="fc1")
            h = tf.matmul(h, self.d_fc1_w) + self.d_fc1_b

            ### summary
            if not reuse:
                tf.summary.histogram("d_fc1_w"   ,self.d_fc1_w)
                tf.summary.histogram("d_fc1_b"   ,self.d_fc1_b)
                tf.summary.histogram("d_conv1_w"   ,self.d_conv1_w)
                tf.summary.histogram("d_conv1_b"   ,self.d_conv1_b)
                tf.summary.histogram("d_conv2_w"   ,self.d_conv2_w)
                tf.summary.histogram("d_conv2_b"   ,self.d_conv2_b)
                tf.summary.histogram("d_conv3_w"   ,self.d_conv3_w)
                tf.summary.histogram("d_conv3_b"   ,self.d_conv3_b)
                tf.summary.histogram("d_conv4_w"   ,self.d_conv4_w)
                tf.summary.histogram("d_conv4_b"   ,self.d_conv4_b)

        return h

    def buildModel(self):
        # define variables
        self.img_dropped  = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 3],name="dropped")
        self.img_original = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 3],name="original")
        self.masks        = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 3],name="masks")

        self.y_real = self.img_original

        self.y_fake  = self.buildGenerator(self.img_dropped)
        self.y_sample = self.buildGenerator(self.img_dropped,reuse=True,isTraining=False)

        self.d_real  = self.buildDiscriminator(self.y_real)
        self.d_fake  = self.buildDiscriminator(self.y_fake,reuse=True)

        # define loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real,labels=tf.ones_like (self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,labels=tf.zeros_like(self.d_fake)))
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        self.g_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,labels=tf.ones_like (self.d_fake)))
        self.g_loss_same = tf.reduce_mean((self.y_fake*self.masks - self.img_dropped*self.masks)**2)
        self.g_loss      = self.g_loss_real + self.g_loss_same

        # define optimizer
        self.g_optimizer = tf.train.AdamOptimizer(self.learnRate,beta1=0.5).minimize(self.g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learnRate,beta1=0.5).minimize(self.d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

        ### summary
        tf.summary.scalar("d_loss_real"   ,self.d_loss_real)
        tf.summary.scalar("d_loss_fake"   ,self.d_loss_fake)
        tf.summary.scalar("d_loss"      ,self.d_loss)
        tf.summary.scalar("g_loss_real" ,self.g_loss_real)
        tf.summary.scalar("g_loss_same" ,self.g_loss_same)
        tf.summary.scalar("g_loss"      ,self.g_loss)

        #############################
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.35))
        self.sess = tf.Session(config=config)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)

        return

    def train(self,f_batch):

        def tileImage(imgs):
            d = int(math.sqrt(imgs.shape[0]-1))+1
            h = imgs[0].shape[0]
            w = imgs[0].shape[1]
            r = np.zeros((h*d,w*d,3),dtype=np.float32)
            for idx,img in enumerate(imgs):
                idx_y = int(idx/d)
                idx_x = idx-idx_y*d
                r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
            return r
        
        if self.saveFolder and not os.path.exists(os.path.join(self.saveFolder,"images")):
            os.makedirs(os.path.join(self.saveFolder,"images"))

        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)

        self.loadModel(self.reload)

        step = -1
        start = time.time()
        while True:
            step += 1

            batch_original,batch_dropped,batch_masks = f_batch(self.nBatch)
            #batch_z        = np.random.uniform(-1.,+1.,[self.nBatch,self.zdim]).astype(np.float32)

            # update generator
            _,d_loss,y_fake,y_real,summary = self.sess.run([self.d_optimizer,self.d_loss,self.y_fake,self.y_real,self.summary],feed_dict={self.img_dropped:batch_dropped, self.img_original:batch_original, self.masks:batch_masks})
            _,g_loss                = self.sess.run([self.g_optimizer,self.g_loss],feed_dict={self.img_dropped:batch_dropped, self.masks:batch_masks})

            if step>0 and step%10==0:
                self.writer.add_summary(summary,step)

            if step%100==0:
                print "%6d: loss(D)=%.4e, loss(G)=%.4e; time/step = %.2f sec"%(step,d_loss,g_loss,time.time()-start)
                start = time.time()
                #g_image = self.sess.run(self.y_sample,feed_dict={self.img_dropped:batch_dropped})
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_drop.png"%step),tileImage(batch_dropped)*255.+128.)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_real.png"%step),tileImage(y_real)*255.+128.)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_fake.png"%step),tileImage(y_fake)*255.+128.)
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)
            #if step==0:
            #    cv2.imwrite(os.path.join(self.saveFolder,"images","org_%d.png"%step),tileImage(batch_images)*255.+128.)
