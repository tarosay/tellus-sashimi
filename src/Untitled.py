#!/usr/bin/env python
# coding: utf-8

# # Tellus API利用サンプル
# このnotebookでは、Tellus APIを利用して衛星画像の取得・表示と簡単な画像処理を行います。
# 
# ***注意:***
# * **`~/examples` 内では上書き保存することができません。編集する場合は `~/work` ディレクトリへコピーしてから実行することをおすすめします。**
# * **このnotebookで取得する全てのデータはTellus上での利用に限定されます**

# In[23]:


import numpy as np
from skimage import io, color, img_as_ubyte, filters
import requests
from io import BytesIO
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Tellusから光学画像の取得と表示
# 皇居周辺の光学画像をTellusから取得します。今回はタイル化されたPNG形式のデータを扱います。位置の指定はXYZ方式を採用しています。詳細は [こちら（外部サイト）](https://maps.gsi.go.jp/development/siyou.html) をご覧ください。
# 
# 詳細補足: ALOSという衛星のAVNIR-2と呼ばれるセンサから取得されたデータを扱います。AVNIR-2については [こちら（外部サイト）](https://www.eorc.jaxa.jp/ALOS/about/javnir2.htm) をご覧ください。

# In[24]:


# 定数
## APIのドメイン
URL_DOMAIN = "gisapi.tellusxdp.com"
BEARER_TOKEN = "5bd4e291-ee42-4369-b0ae-3eedd72b737d " # APIトークンをセットしてください

## 皇居周辺の座標
Z = 13
X = 7185
Y = 3287

if BEARER_TOKEN == "":
    print("APIトークンがセットされていません")


# In[25]:


def get_data(img_type, domain=URL_DOMAIN, z=Z, x=X, y=Y, query=""):
    if query != "":
        query = "?" + query

    # AuthorizationヘッダにAPIトークンをセットすることで認証されます
    res = requests.get("https://{}/{}/{}/{}/{}.png{}".format(URL_DOMAIN, img_type, z, x, y, query),
                   headers={"Authorization": "Bearer " + BEARER_TOKEN})

    # 取得した画像データをNumPyの多次元配列形式で保持します
    img = img_to_array(load_img(BytesIO(res.content)))
    return img.astype(np.uint8)

img_osm = get_data("osm")
img_band1= get_data("blend", query="r=3&g=2&b=1&rdepth=0&gdepth=0")
img_band2 = get_data("blend", query="r=3&g=2&b=1&bdepth=0&rdepth=0")
img_band3 = get_data("blend", query="r=3&g=2&b=1&gdepth=0&bdepth=0")
img_band4 = get_data("blend", query="r=4&g=4&b=4")
img_ndvi = get_data("blend", query="preset=ndvi")


# ### OpenStreetMap

# In[26]:


print(img_osm.shape)
io.imshow(img_osm)


# ### AVNIR-2
# AVNIR-2では4つの異なる波長のデータを利用できます。大まかに、Band1は青の波長、Band2は緑の波長、Band3は赤の波長、Band4は近赤外の波長に対応します。それぞれ単独で見ると以下のような画像になります。

# In[27]:


io.imshow(np.hstack((img_band1, img_band2, img_band3, img_band4)))


# #### True Color合成
# それぞれ可視光の青・緑・赤・（近赤外）に対応しているため、RGB（赤緑青）として1枚の画像に合成すると人の目で見た色に近いものになります。このようにして合成されたものTrue Color画像と呼びます。
# 
# True ColorのRGB合成
# * R: Band3（赤）
# * G: Band2（緑）
# * B: Band1（青）

# In[28]:


img_true = np.c_[img_band3[:,:,0:1], img_band2[:,:,1:2], img_band1[:,:,2:3]]
io.imshow(img_true)


# #### Natural Color合成
# True Color以外にも合成方法はあり、例えば植生域だけを際立たせたい時は、植物の分布域が緑で表現されるNatural Colorと呼ばれる合成が用いられます。これは、近赤外線は植物の反射率が高いことを利用し、RGBの赤に赤の波長であるBand3、緑に近赤外の波長であるBand4、青に緑の波長であるBand2を割り当てたものです。
# 
# Natural ColorのRGB合成
# * R: Band3（赤）
# * G: Band4（近赤外）
# * B: Band2（緑）

# In[29]:


img_natural = np.c_[img_band3[:,:,0:1], img_band4[:,:,0:1], img_band2[:,:,1:2]]
io.imshow(img_natural)


# In[ ]:





# 皇居周辺に植物が多いことが見て取れます。

# #### NDVI
# Normalized Difference Vegetation Index (NDVI) を用いることでより植生を見ることができます。NDVIとは正規化植生指標のことであり、可視域赤 ($R$, Band3) と近赤外 ($IR$, Band4) のデータから以下の式で計算されます。$[-1, 1]$ に正規化され、値が大きいほど植生が多いことを表します。
# 
# $$
# NDVI = \frac{IR - R}{IR + R}
# $$

# In[30]:


io.imshow(img_ndvi)


# In[ ]:





# ### グレースケール化
# #### グレースケール化①
# `skimage.color.rgb2gray` を使って直接グレースケール化を試みます。

# In[31]:


# カラー画像からGrayscale画像への変換
img_gray_01 = color.rgb2gray(img_true)

# 値のレンジを変更 ([0, 1] -> [0, 255])
img_gray = img_as_ubyte(img_gray_01) 

print("変換前: [0, 1]")
print(img_gray_01)
print("変換後: [0, 255]")
print(img_gray)

io.imshow(img_gray.astype(np.uint8))


# #### グレースケール化②
# 別のグレースケール化方法も試してみましょう。一度RGB空間からYIQ空間へ変換し、Yを利用を利用します。YIQ形式は、グレースケール情報がカラーデータから分離しているため、同じ信号をカラーと白黒の両方で使用可能です。
# 
# （※ グレースケール化のアルゴリズムによっては `img_gray_01` と `img_yiq[:, :, 0]` が等しくなりますが、skimageでは異なります）

# In[38]:


img_yiq = color.rgb2yiq(img_true)
img_conb = np.concatenate(
    (img_yiq[:, :, 0], img_yiq[:, :, 1], img_yiq[:, :, 2]), axis=1)
io.imshow(img_conb)


# In[33]:


# skimage.color.rgb2gray と比較
img_conb2 = np.concatenate((img_yiq[:, :, 0], img_gray_01), axis=1)
io.imshow(img_conb2)


# In[47]:


# 反転画像も確認
img_nega = 255 - img_gray
io.imshow(img_nega)
#io.imshow(img_gray)
print(img_nega)
np.savetext('AAA.txt',img_nega,delimiter=',')


# #### グレースケール化されたデータの統計情報の確認

# In[45]:


print('pixel sum', np.sum(img_gray[:, :]))
print('pixel mean', np.mean(img_gray[:,:]))
print('pixel variance', np.var(img_gray[:,:]))
print('pixel stddev', np.std(img_gray[:,:]))


# #### ヒストグラムの確認

# In[36]:


hists, bins = np.histogram(img_gray, 255, [0, 255])
plt.plot(hists)


# In[22]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'Untitled.ipynb'])


# In[ ]:




