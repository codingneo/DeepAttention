from keras.layers.recurrent import GRU

class FeedForwardGRU(GRU):
	def __init__(self, **kwargs):
		super(GRU, self).__init__(kwargs)

	def build():
		super(GRU, self).build()
		self.W_y = self.init((self.hidden_dim, self.output_dim), name='{}_W_y'.format(self.name))
        self.b_y = K.zeros((self.output_dim), name='{}_b_y'.format(self.name))
        self.trainable_weights += [self.W_y, self.b_y]

    def step(self, x, states):
    	states = list(states)
        y_pre = states.pop(2)
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
    	h_t, states = super(GRU, self).step(y_pre, states)
    	self.output_dim = output_dim
    	y_t = self.activation(K.dot(h_t, self.W_y) + self.b_y)
    	new_states += [y_t]
    	return h_t, new_states

    def call(self, x, mask=None):
    	input_shape = self.input_spec[0].shape
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)
        initial_states += []
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output


