import tensorflow as tf

model = tf.keras.models.load_model(
    "model/mo_hinh_nhan_dien_khuon_mat.h5",
    compile=False
)

model.summary()
