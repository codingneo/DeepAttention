from keras.layers.recurrent import GRU

class FeedForwardGRU(GRU):
	def __init__(self, **kwargs):
		super(GRU, self).__init__(kwargs)
