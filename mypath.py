class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return '/four_tb/Cityscapes/leftImg8bit_trainvaltest/'     # folder that contains leftImg8bit/
        elif dataset == 'rainy_cityscapes':
            return '/four_tb/Perception/leftImg8bit_trainvaltest/'
        elif dataset == 'foggy_cityscapes':
            return '/four_tb/Foggy/leftImg8bit_trainvaltest/'
        elif dataset == 'GTAV':
            return '/four_tb/GTAV/'
        elif dataset == 'GTAV_Synthia':
            return '/four_tb/GTAV_for_Synthia/'
        elif dataset == 'BDD100k':
            return '/four_tb/BDD100k/bdd100k/seg/'
        elif dataset == 'SYNTHIA':
            return '/four_tb/RAND_CITYSCAPES/'
        elif dataset == 'Mapillary':
            return '/four_tb/Mappillary/'
        elif dataset == 'M3FD':
            return '/four_tb/m3fd/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError