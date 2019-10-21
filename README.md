# image_search
Image search with tf.2.x


# example
```python
from image_search import ImageSearch

fe = ImageSearch()
fe.reset_feature_folder()
fe.save_feature('C:/Users/JinqingLee/Desktop/image')
fe.load_feature()
scores = fe.search('C:/Users/JinqingLee/Desktop/image/14903831_1200x1000_0.jpg')

import pyecharts as pe
def image_base(img_src, title, subtitle):
    image = (pe.components.Image()
             .add(src=img_src)
             .set_global_opts(title_opts=pe.options.ComponentTitleOpts(title=title, subtitle=subtitle)))
    return image
image_list = []
for i in scores:
    image_list.append(image_base(i[1], str(i[0].numpy().tolist()), ''))
page = pe.charts.Page(layout=pe.charts.Page.SimplePageLayout).add(*image_list)
page.render()
```
