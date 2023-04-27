# image_captioning
Pembuatan hasil deskripsi gambar menggunakan reinforcement learning dan reference-based long short term memory

Main file : simple_caption.ipynb <br>

Dataset menggunakan Flick8k dataset yang dapat didownload dari : <br>
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip <br>
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip <br>

File yang terlibat : description.txt, features.p, tokenizer.p <br>

Program akan melakukan pengelompokan dataset serta mengesktrak fitur pada gambar dataset yang disimpan dalam "features.p"<br>
Program menggunakan model LSTM untuk melakukan pelatihan model dengan dataset_Train yang telah dikelompokkan.<br>

Error : Program terhenti di tengah pelatihan model pada sesi pelatihan dengan error **TypeError: `generator` yielded an element of shape (0,) where an element of shape (None, None) was expected.** <br>
Error tersebut terjadi pada line **model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)**
