# coding = utf-8

class Sentence(object):
    """ A Sentence Data Class. """
    def __init__(self, sen_tagged, wsj_section):
        """ basic info """
        # sen_tagged with BIOEST, for example: Not/RB/O only/RB/O ... .
        self.sen_tagged = sen_tagged
        # the wsj section the sentence come from, for example: wsj.section00.
        self.wsj_section = wsj_section
        # the sentence, for example: Not only ... .
        self.sen = None 
        # show whether the sentence is gold or generated
        self.is_generated = False
        # a list of words in the sentence, for example: ['Not', 'only', ...].
        self.words_list = []
        # a list of POS index(refer to POS_2_INDEX below) of the words in the sentence, for example: [19, 19, ...].
        self.pos_list = []
        # a list of tag index(refer to TAG_2_INDEX below) of the words in the sentence, for example: [2, 2, ...].
        self.tag_list = []

        """ for trigger training """
        # the trigger word in the sentence, for example: is.
        self.sen_trigger_word = None
        # the auxiliay type of the trigger, for example: be.
        self.sen_trigger_aux_type = None
        # the training input vector
        self.sen_vec = None
        # the training input vector with feature
        self.sen_vec_feature = None
        # tag (1 for positive case, 0 for negative case), for example: 1
        self.trigger_label = None

        """ for antecedent training """
        # the berkely parse result of the sentence
        self.sen_parse = None
        # the trigger index in the words list
        self.trigger_index = None
        # list which marked the gold antecedent with 1, otherwise 0
        self.truth_label = None
        # words before antecedent
        self.words_before_antecedent = None
        # antecedent words
        self.antecedent = None
        # words after antecedent
        self.words_after_antecedent = None
        # list which marked the words with 0(other), 1(antecedent) and 2(trigger) 
        self.sen_tag_label = None
        
        # hidden
        # GRU hidden, Tensor, Size: sentences_length X 6000
        self.before_antecedent_hidden_tensor = None
        self.antecedent_hidden_tensor = None
        self.after_antecedent_hidden_tensor = None

        # attention
        # GRU hidden, Tensor, Size: sentences_length X 1500
        self.before_antecedent_attention_tensor = None
        self.antecedent_attention_tensor = None
        self.after_antecedent_attention_tensor = None

        # input vec 
        # sum pooling, FloatTensor, Size: 400
        self.input_vec_sum = None
        # sum pooling with feature, FloatTensor, Size: 468
        self.input_vec_sum_feature = None
        # GRU, FloatTensor, Size: 6100
        self.input_vec_hidden = None
        # GRU with feature, FloatTensor, Size: 6168
        self.input_vec_hidden_feature = None
        # AttentionGRU, FloatTensor, Size: 1600
        self.input_vec_attention = None
        # AttentionGRU with feature, FloatTensor, Size: 1668
        self.input_vec_attention_feature = None
        # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        self.antecedent_label = None