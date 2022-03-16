import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

Flatten_Norm_Testing_Images = np.load("Flatten_Norm_Testing_Images.npy")
Flatten_Norm_Training_Images = np.load("Flatten_Norm_Training_Images.npy")
Training_Labels = np.load("Training_Labels.npy")
Testing_Labels = np.load("Testing_Labels.npy")

feat_cols = ['pixel' + str(i) for i in range(Flatten_Norm_Training_Images.shape[1])]
df_attendance = pd.DataFrame(Flatten_Norm_Training_Images, columns=feat_cols)

df_attendance['label'] = Training_Labels

pca_attendance = PCA(n_components=8)
principalComponents_attendance = pca_attendance.fit_transform(df_attendance.iloc[:, :-1])

principal_attendance_Df = pd.DataFrame(data=principalComponents_attendance,
                                       columns=['principal component 1', 'principal component 2',
                                                'principal component 3',
                                                'principal component 4',
                                                'principal component 5', 'principal component 6',
                                                'principal component 7', 'principal component 8',
                                                ])
principal_attendance_Df['y'] = Training_Labels

# plt.figure(figsize=(16, 4))  # change with apo
# sns.scatterplot(
#    x="principal component 1", y="principal component 2",
#    hue="y",
#    palette=sns.color_palette("hls", 4),  # change to 4
#    data=principal_attendance_Df,
#    legend="full",
#    alpha=1
# )

pca = PCA(0.95)

pca.fit(Flatten_Norm_Training_Images)

train_img_pca = pca.transform(Flatten_Norm_Training_Images)
test_img_pca = pca.transform(Flatten_Norm_Testing_Images)
