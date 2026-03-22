import numpy as np
import os

SPATIAL_FEATURES = ['cdp_x.npy', 'cdp_y.npy', 'twt.npy']
GEOM_FEATURES = [
    'loc_struct_azim_cos.npy',
    'loc_struct_azim_sin.npy',
    'dip_dev.npy',
    'dip_usual.npy',
]
SPECTRAL_FEATURES = ['2019_15Hz.npy', '2019_30Hz.npy', '2019_45Hz.npy']
ATTRIBUTE_FEATURES = [
    '2019_15Hz.npy',
    '2019_30Hz.npy',
    '2019_45Hz.npy',
    '2019_SUMM.npy',
    'FF.npy',
    'offset_0950.npy',
    'offset_1600.npy',
    'offset_2500.npy',
    'offset_ENV1600.npy',
    'offset_ENV2500.npy',
    'offset_ENV950.npy',
    'loc_struct_azim_cos.npy',
    'loc_struct_azim_sin.npy',
    'dip_dev.npy',
    'dip_usual.npy',
]
ALL_STAGE1_FEATURES = SPATIAL_FEATURES + ATTRIBUTE_FEATURES

class TwoStageClust:

    def __init__(self, data_folder,
                 n_initial_clusters=20000):
        # self.rooted_alg = rooted_alg
        self.n_initial_clusters = n_initial_clusters
        # self.used_features = used_features
        
        self.data_folder = data_folder
    
    def load_features(self, use_spatial, attrib_config, 
                      weights, feature_files=None):
        
        feat_names = self.take_features(use_spatial, 
                                        attrib_config,
                                        feature_files=feature_files,
                                        )
        print(f'used features: {feat_names}')
        features = []
        for feat in feat_names:
            feat_npy = np.load(os.path.join(self.data_folder, 
                                            feat))
            feat_npy = self.scale_features(feat_npy)
            if weights is not None:
                feat_npy = (feat_npy * weights[feat] if feat in 
                            weights.keys() else feat_npy)
            features.append(feat_npy[:, None])
        self.features = np.concatenate(features, axis=1)
        print(f'Features was loaded successfully!')
        
    def scale_features(self, x):
        return (x - x.min()) / (x.max() - x.min())


    def take_features(self, use_spatial, attrib_config, feature_files=None):
        if feature_files is not None:
            feat_used = []
            if use_spatial == 'all':
                feat_used.extend(SPATIAL_FEATURES)
            elif use_spatial == 'only_twt':
                feat_used.append(SPATIAL_FEATURES[-1])
            feat_used.extend(feature_files)
            return list(dict.fromkeys(feat_used))

        feat_used = []
        if use_spatial == 'all':
            feat_used.extend(SPATIAL_FEATURES)

        elif use_spatial=='only_twt':
            feat_used.append(SPATIAL_FEATURES[-1])
            # feat_used.remove(feat_spat[0])
            # feat_used.remove(feat_spat[1])
        # else:

        if attrib_config == 'only_spectr':
            feat_used.extend(SPECTRAL_FEATURES)
            return feat_used
        elif attrib_config=='no_spectr':
            feat_used.extend(ATTRIBUTE_FEATURES)
            for sp_f in SPECTRAL_FEATURES:
                feat_used.remove(sp_f)
            return feat_used
        elif attrib_config=='only_geom':
            feat_used.extend(GEOM_FEATURES)
            return feat_used
        else:
            feat_used.extend(ATTRIBUTE_FEATURES)
            return feat_used
    def take_feat_as_list(self, feat_used):
        return feat_used

    def fit_predict(self, rooted_alg, use_kmeans_centr=True):
        self.rooted_alg = rooted_alg
        
        if use_kmeans_centr:
            from fastkmeans import FastKMeans

            print(f'Use two stage clustering with {self.n_initial_clusters} centroids')
            d = self.features.shape[1]
            k_means = FastKMeans(d=d,
                            k=self.n_initial_clusters, use_triton=False
                            )

            kmeans_labels = k_means.fit_predict(self.features)

            cluster_centers = k_means.centroids
            # cluster_sizes = np.bincount(kmeans_labels)

            # weighted_centers = []
            # for i, center in enumerate(cluster_centers):
            #     for _ in range(int(np.log(cluster_sizes[i] + 1))):
            #         jittered_center = center + np.random.normal(0, 0.001, center.shape)
            #         weighted_centers.append(jittered_center)

           
            # dbscan_reduced = DBSCAN(eps=0.3, min_samples=3)
            # gmm = GaussianMixture(n_components=5)
            reduced_labels = self.rooted_alg.fit_predict(cluster_centers)

            final_labels = np.array([reduced_labels[label] for label in kmeans_labels])
        else:
            print('Use picked algorithm directly')
            final_labels = self.rooted_alg.fit_predict(self.features)
        return final_labels
