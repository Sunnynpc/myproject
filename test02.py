from PIL import Image
import numpy as np
# 打开图片
img = Image.open("data/2007_000129.jpg")
# 得到图片的宽、高
w,h = img.size
# 得到宽、高中更大的一个
side = max(w,h)
# 得到缩放比例l
l = 225/side
# 输出图片缩放后尺寸
# print(int(w*l),int(h*l))
# 对图片在不变形的前提下进行缩放
img = img.resize((int(w*l),int(h*l)))
# 创建一个全黑的背景图片
bg = np.zeros((225,225,3),dtype=np.uint8)
bg_img = Image.fromarray(bg,"RGB")

# bg_img.paste(img,(((225-int(w*l))//2),0))
# 将缩放后的图片粘贴在背景图片上
bg_img.paste(img)
# 展示图片
bg_img.show()