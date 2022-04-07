def sequence_selection(B, S, linkage_threshold):
    '''

    :param B: a set of region propsoal for an entire clip (num frames, num boxes, 4)
    :param S: a set of scores for the entire clip (num frames, num boxes)
    :param linkage_theshold:
    :return: a dictionary
    '''
