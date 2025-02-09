import os
import io
import shutil
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask_socketio import SocketIO, emit
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Initialize SocketIO using threading mode
socketio = SocketIO(app, async_mode='threading')

# Directories for image uploads and model storage
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Allow up to 1 GB uploads
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024

def create_model(input_shape, learning_rate, num_conv_layers, conv_filters,
                 kernel_size, stride, padding, dense_neurons, num_classes):
    """
    Build a CNN model.
    - For binary classification (num_classes == 2), uses a single neuron with sigmoid activation.
    - For multi-class classification (num_classes > 2), uses num_classes neurons with softmax activation.
    """
    model = Sequential()
    for i in range(num_conv_layers):
        filters = conv_filters[i]
        if i == 0:
            model.add(Conv2D(filters=filters,
                             kernel_size=(kernel_size, kernel_size),
                             strides=(stride, stride),
                             padding=padding,
                             activation='relu',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(filters=filters,
                             kernel_size=(kernel_size, kernel_size),
                             strides=(stride, stride),
                             padding=padding,
                             activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for neurons in dense_neurons:
        model.add(Dense(neurons, activation='relu'))
    
    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------
# Route to set classes dynamically
# ---------------------------
@app.route('/set_classes', methods=['GET', 'POST'])
def set_classes():
    if request.method == 'POST':
        class_names = request.form.get('class_names')
        if class_names:
            # Store the list of classes in the session using the key "classes"
            session['classes'] = [name.strip() for name in class_names.split(',') if name.strip()]
            flash("Classes set successfully!")
            return redirect(url_for('upload_images'))
        else:
            flash("Please enter at least one class name.")
    return render_template('set_classes.html')

# ---------------------------
# Upload images route
# ---------------------------
@app.route('/upload_images')
def upload_images():
    # Retrieve class names from the session using the key "classes"
    class_names = session.get('classes', [])
    return render_template('upload_images.html', class_names=class_names)

# ---------------------------
# Endpoint to upload a single image
# ---------------------------
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file part", 400
    image_file = request.files['image']
    if image_file.filename == '':
        return "No file selected", 400
    # Retrieve the class for this image from the form (field "img_class")
    img_class = request.form.get('img_class')
    if not img_class:
        return "No class specified", 400
    target_folder = os.path.join(app.config['UPLOAD_FOLDER'], img_class)
    os.makedirs(target_folder, exist_ok=True)
    filename = secure_filename(image_file.filename)
    image_file.save(os.path.join(target_folder, filename))
    return "File uploaded successfully", 200

# ---------------------------
# Start training route with live progress updates
# ---------------------------
@app.route('/start_training', methods=['GET', 'POST'])
def start_training():
    if request.method == 'POST':
        # Retrieve training hyperparameters
        learning_rate = float(request.form.get('learning_rate', 0.001))
        batch_size = int(request.form.get('batch_size', 32))
        epochs = int(request.form.get('epochs', 10))
        kernel_size = int(request.form.get('kernel_size', 3))
        stride = int(request.form.get('stride', 1))
        padding = request.form.get('padding', 'same')
        num_conv_layers = int(request.form.get('num_conv_layers', 1))
        
        conv_filters_str = request.form.get('conv_filters', '32')
        conv_filters = [int(x.strip()) for x in conv_filters_str.split(',') if x.strip()]
        if len(conv_filters) != num_conv_layers:
            flash("Number of convolution filters does not match the number of conv layers.")
            return redirect(url_for('start_training'))
        
        dense_neurons_str = request.form.get('dense_neurons', '')
        dense_neurons = [int(x.strip()) for x in dense_neurons_str.split(',') if x.strip()] if dense_neurons_str else []
        
        # Retrieve the custom classes from session; default to two classes if not set.
        classes = session.get('classes', [])
        if len(classes) < 2:
            classes = ["class1", "class2"]
        num_classes = len(classes)
        
        # Set class_mode based on number of classes.
        class_mode = 'categorical' if num_classes > 2 else 'binary'
        
        target_size = (64, 64)
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        # Pass the list of classes to ensure the generator uses exactly these folders.
        train_generator = datagen.flow_from_directory(
            app.config['UPLOAD_FOLDER'],
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset='training',
            classes=classes
        )
        validation_generator = datagen.flow_from_directory(
            app.config['UPLOAD_FOLDER'],
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset='validation',
            classes=classes
        )
        
        input_shape = (target_size[0], target_size[1], 3)
        model = create_model(input_shape, learning_rate, num_conv_layers, conv_filters,
                             kernel_size, stride, padding, dense_neurons, num_classes)
        
        steps_per_epoch = train_generator.samples // batch_size
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        validation_steps = validation_generator.samples // batch_size
        if validation_steps < 1:
            validation_steps = 1
        
        # Custom callback to emit live training progress via Socket.IO.
        class SocketIOCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                print(f"Epoch {epoch+1} complete. Logs: {logs}")
                socketio.emit('epoch_update', {
                    'epoch': epoch + 1,
                    'train_accuracy': logs.get('accuracy', 0),
                    'val_accuracy': logs.get('val_accuracy', 0)
                })
        
        def background_train():
            with app.app_context():
                print("Background training started")
                history = model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[SocketIOCallback()]
                )
                print("Background training finished")
                model_path = os.path.join(MODEL_FOLDER, 'cnn_model.h5')
                model.save(model_path)
                # Delete the entire uploads folder and recreate it.
                if os.path.exists(app.config['UPLOAD_FOLDER']):
                    shutil.rmtree(app.config['UPLOAD_FOLDER'])
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                socketio.emit('training_complete', {'message': 'Training complete!'})
        
        socketio.emit('training_started', {'message': 'Training started...'})
        socketio.start_background_task(background_train)
        return render_template('training_progress.html')
    return render_template('start_training.html')

# ---------------------------
# Prediction route (supports multi-class)
# ---------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_path = os.path.join(MODEL_FOLDER, 'cnn_model.h5')
    if not os.path.exists(model_path):
        flash("Model not found. Please train a model first!")
        return redirect(url_for('index'))
    model = tf.keras.models.load_model(model_path)
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            target_size = (64, 64)
            image = Image.open(filepath).convert('RGB')
            image = image.resize(target_size)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            prediction = model.predict(image_array)
            # Retrieve custom classes from session; default to binary if not set.
            classes = session.get('classes', ["class1", "class2"])
            if len(classes) < 2:
                classes = ["class1", "class2"]
            # If more than two classes, use np.argmax; else, threshold at 0.5.
            if len(classes) > 2:
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_class = classes[predicted_index]
            else:
                predicted_class = classes[1] if prediction[0][0] > 0.5 else classes[0]
            flash(f"Predicted class: {predicted_class}")
            return redirect(url_for('predict'))
    return render_template('predict.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)
