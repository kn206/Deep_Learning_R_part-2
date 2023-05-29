{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e803ecf2",
   "metadata": {
    "papermill": {
     "duration": 0.005604,
     "end_time": "2023-05-29T16:53:25.505213",
     "exception": false,
     "start_time": "2023-05-29T16:53:25.499609",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# \"Deep Learning Modeling in R: From Definition to Prediction\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37916b14",
   "metadata": {
    "papermill": {
     "duration": 0.004464,
     "end_time": "2023-05-29T16:53:25.514510",
     "exception": false,
     "start_time": "2023-05-29T16:53:25.510046",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "1. **Model Definition:**\n",
    "   - A deep learning model architecture is defined using the Keras Sequential API.\n",
    "   - The model consists of multiple layers, including convolutional, pooling, and dense layers, to capture meaningful features from the input data.\n",
    "\n",
    "2. **Model Compilation:**\n",
    "   - The model is compiled using the `compile()` function.\n",
    "   - The loss function, optimizer, and evaluation metric(s) are specified.\n",
    "   - This step prepares the model for training by configuring the necessary components for optimization and evaluation.\n",
    "\n",
    "3. **Model Training:**\n",
    "   - The model is trained on the training data using the `fit()` function.\n",
    "   - The training process iterates over the specified number of epochs, adjusting the model's parameters to minimize the defined loss function.\n",
    "   - The training data is fed in batches, and the model's performance is monitored on a validation set.\n",
    "\n",
    "4. **Model Evaluation:**\n",
    "   - The trained model is evaluated on the test data using the `evaluate()` function.\n",
    "   - Evaluation metrics, such as accuracy or loss, are calculated to measure the model's performance on unseen data.\n",
    "\n",
    "5. **Model Prediction:**\n",
    "   - The trained model is used to make predictions on new, unseen data using the `predict()` function.\n",
    "   - The model's output provides predictions or probabilities for the target variable(s) based on the given input.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1125edf3",
   "metadata": {
    "papermill": {
     "duration": 0.00442,
     "end_time": "2023-05-29T16:53:25.523615",
     "exception": false,
     "start_time": "2023-05-29T16:53:25.519195",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Lets do this step by step:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "043f685b",
   "metadata": {
    "papermill": {
     "duration": 0.004408,
     "end_time": "2023-05-29T16:53:25.532808",
     "exception": false,
     "start_time": "2023-05-29T16:53:25.528400",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**1. Loading Libraries**\n",
    "The necessary libraries, including keras and tensorflow, are loaded to provide the required functionalities for working with deep learning models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "45487c13",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:53:25.546121Z",
     "iopub.status.busy": "2023-05-29T16:53:25.543966Z",
     "iopub.status.idle": "2023-05-29T16:53:26.900360Z",
     "shell.execute_reply": "2023-05-29T16:53:26.898594Z"
    },
    "papermill": {
     "duration": 1.36579,
     "end_time": "2023-05-29T16:53:26.903380",
     "exception": false,
     "start_time": "2023-05-29T16:53:25.537590",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load required libraries\n",
    "library(keras)\n",
    "library(tensorflow)\n",
    "library(ggplot2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac0f70e0",
   "metadata": {
    "papermill": {
     "duration": 0.004623,
     "end_time": "2023-05-29T16:53:26.913046",
     "exception": false,
     "start_time": "2023-05-29T16:53:26.908423",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**2.Loading the Data **\n",
    "\n",
    "The MNIST dataset is loaded using the dataset_mnist() function from the keras library. It retrieves the MNIST dataset, which consists of handwritten digits images and their corresponding labels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4967edb1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:53:26.973912Z",
     "iopub.status.busy": "2023-05-29T16:53:26.924234Z",
     "iopub.status.idle": "2023-05-29T16:53:37.102525Z",
     "shell.execute_reply": "2023-05-29T16:53:37.100862Z"
    },
    "papermill": {
     "duration": 10.187886,
     "end_time": "2023-05-29T16:53:37.105723",
     "exception": false,
     "start_time": "2023-05-29T16:53:26.917837",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load MNIST dataset\n",
    "mnist <- dataset_mnist()\n",
    "x_train <- mnist$train$x\n",
    "y_train <- mnist$train$y\n",
    "x_test <- mnist$test$x\n",
    "y_test <- mnist$test$y\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cdc280c3",
   "metadata": {
    "papermill": {
     "duration": 0.004787,
     "end_time": "2023-05-29T16:53:37.116517",
     "exception": false,
     "start_time": "2023-05-29T16:53:37.111730",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**3.Preprocessing**\n",
    "\n",
    "The test set images x_test are reshaped to have a 4D tensor shape of (num_samples, 28, 28, 1). This is required because the MNIST images are grayscale and have dimensions of 28x28 pixels.\n",
    "The pixel values in x_test are normalized by dividing them by 255 to scale the values between 0 and 1. This step is essential for effective training and prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "caf75624",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:53:37.128413Z",
     "iopub.status.busy": "2023-05-29T16:53:37.127196Z",
     "iopub.status.idle": "2023-05-29T16:53:38.610296Z",
     "shell.execute_reply": "2023-05-29T16:53:38.608607Z"
    },
    "papermill": {
     "duration": 1.492212,
     "end_time": "2023-05-29T16:53:38.613255",
     "exception": false,
     "start_time": "2023-05-29T16:53:37.121043",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Preprocess the data\n",
    "x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))\n",
    "x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))\n",
    "x_train <- x_train / 255\n",
    "x_test <- x_test / 255"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "75efec38",
   "metadata": {
    "papermill": {
     "duration": 0.004605,
     "end_time": "2023-05-29T16:53:38.622844",
     "exception": false,
     "start_time": "2023-05-29T16:53:38.618239",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**4.Defining model Architecture**\n",
    "\n",
    "The trained model is loaded using the load_model_hdf5() function from the keras library. The function takes the path to the saved model file in HDF5 format and loads the model into memory. The loaded model contains the architecture, weights, and other necessary parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ff1300f9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:53:38.634863Z",
     "iopub.status.busy": "2023-05-29T16:53:38.633612Z",
     "iopub.status.idle": "2023-05-29T16:53:41.543182Z",
     "shell.execute_reply": "2023-05-29T16:53:41.541455Z"
    },
    "papermill": {
     "duration": 2.918533,
     "end_time": "2023-05-29T16:53:41.546091",
     "exception": false,
     "start_time": "2023-05-29T16:53:38.627558",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Define the model architecture\n",
    "model <- keras_model_sequential()\n",
    "model %>%\n",
    "  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = \"relu\", input_shape = c(28, 28, 1)) %>%\n",
    "  layer_max_pooling_2d(pool_size = c(2, 2)) %>%\n",
    "  layer_dropout(0.25) %>%\n",
    "  layer_flatten() %>%\n",
    "  layer_dense(units = 128, activation = \"relu\") %>%\n",
    "  layer_dropout(0.5) %>%\n",
    "  layer_dense(units = 10, activation = \"softmax\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e531dd71",
   "metadata": {
    "papermill": {
     "duration": 0.004631,
     "end_time": "2023-05-29T16:53:41.555634",
     "exception": false,
     "start_time": "2023-05-29T16:53:41.551003",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**5.Compiling the Model**\n",
    "\n",
    "I compile the model using the compile() function. This step defines the loss function, optimizer, and metrics to be used during training.\n",
    "I specify \"categorical_crossentropy\" as the loss function, which is suitable for multi-class classification problems like the MNIST dataset.\n",
    "I use the optimizer_adam() function to define the optimizer. Adam is a popular optimization algorithm for deep learning models.\n",
    "I include \"accuracy\" in the metrics list to track the accuracy of the model during training and evaluation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "890a43b9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:53:41.567723Z",
     "iopub.status.busy": "2023-05-29T16:53:41.566329Z",
     "iopub.status.idle": "2023-05-29T16:53:41.593582Z",
     "shell.execute_reply": "2023-05-29T16:53:41.592080Z"
    },
    "papermill": {
     "duration": 0.035782,
     "end_time": "2023-05-29T16:53:41.596053",
     "exception": false,
     "start_time": "2023-05-29T16:53:41.560271",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Compile the model\n",
    "optimizer <- tf$keras$optimizers$Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = FALSE)\n",
    "model %>% compile(\n",
    "  loss = \"sparse_categorical_crossentropy\",\n",
    "  optimizer = optimizer,\n",
    "  metrics = c(\"accuracy\")\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "425848cb",
   "metadata": {
    "papermill": {
     "duration": 0.004546,
     "end_time": "2023-05-29T16:53:41.605481",
     "exception": false,
     "start_time": "2023-05-29T16:53:41.600935",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**6.Training the Model**\n",
    "\n",
    "I use the fit() function to train the model on the training data x_train and y_train.\n",
    "The model is trained for 10 epochs, with a batch size of 128, and a validation split of 0.2.\n",
    "The training process adjusts the model's parameters to minimize the defined loss function and improve its performance on the training data.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "03611209",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:53:41.617461Z",
     "iopub.status.busy": "2023-05-29T16:53:41.616224Z",
     "iopub.status.idle": "2023-05-29T16:54:07.695674Z",
     "shell.execute_reply": "2023-05-29T16:54:07.693875Z"
    },
    "papermill": {
     "duration": 26.08877,
     "end_time": "2023-05-29T16:54:07.698943",
     "exception": false,
     "start_time": "2023-05-29T16:53:41.610173",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "# Train the model\n",
    "history <- model %>% fit(\n",
    "  x_train, y_train,\n",
    "  epochs = 10,\n",
    "  batch_size = 128,\n",
    "  validation_split = 0.2\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a2bb655",
   "metadata": {
    "papermill": {
     "duration": 0.004747,
     "end_time": "2023-05-29T16:54:07.708966",
     "exception": false,
     "start_time": "2023-05-29T16:54:07.704219",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**7.Evaluating the Model**\n",
    "\n",
    "I use the evaluate() function to evaluate the model's performance on the test set x_test and y_test.\n",
    "The evaluation result is stored in the evaluation variable, which contains metrics such as loss and accuracy.\n",
    "I extract the accuracy value from the evaluation result using evaluation[[2]] and store it in the accuracy variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "29f8ea38",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:54:07.721658Z",
     "iopub.status.busy": "2023-05-29T16:54:07.720374Z",
     "iopub.status.idle": "2023-05-29T16:54:08.712585Z",
     "shell.execute_reply": "2023-05-29T16:54:08.710157Z"
    },
    "papermill": {
     "duration": 1.001162,
     "end_time": "2023-05-29T16:54:08.715308",
     "exception": false,
     "start_time": "2023-05-29T16:54:07.714146",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.9862 \n"
     ]
    }
   ],
   "source": [
    "# Evaluate the model on the test set\n",
    "evaluation <- model %>% evaluate(x_test, y_test)\n",
    "accuracy <- evaluation[[2]]\n",
    "cat(\"Accuracy: \", accuracy, \"\\n\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dcc91fba",
   "metadata": {
    "papermill": {
     "duration": 0.004733,
     "end_time": "2023-05-29T16:54:08.725001",
     "exception": false,
     "start_time": "2023-05-29T16:54:08.720268",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**8.Printing the History**\n",
    "\n",
    "Finally, I print the obtained accuracy value using the cat() function. The accuracy represents the performance of the trained model on the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9975ffec",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:54:08.737094Z",
     "iopub.status.busy": "2023-05-29T16:54:08.735855Z",
     "iopub.status.idle": "2023-05-29T16:54:08.752205Z",
     "shell.execute_reply": "2023-05-29T16:54:08.750694Z"
    },
    "papermill": {
     "duration": 0.024825,
     "end_time": "2023-05-29T16:54:08.754609",
     "exception": false,
     "start_time": "2023-05-29T16:54:08.729784",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Final epoch (plot to see history):\n",
      "        loss: 0.05434\n",
      "    accuracy: 0.9836\n",
      "    val_loss: 0.04558\n",
      "val_accuracy: 0.9862 \n"
     ]
    }
   ],
   "source": [
    "print(history)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5165ac6a",
   "metadata": {
    "papermill": {
     "duration": 0.004682,
     "end_time": "2023-05-29T16:54:08.764260",
     "exception": false,
     "start_time": "2023-05-29T16:54:08.759578",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "A deep learning model's architecture and layer configurations are specified using Keras during the modeling process. The model is then assembled using the optimizer, loss function, and evaluation metric(s) of choice. It is trained using the training data and its parameters are incrementally changed over several epochs to minimize the loss function. The test data are used to assess the performance of the trained model by calculating metrics like accuracy and loss. Finally, using the discovered relationships and patterns as inputs, the trained model is used to forecast new data. In general, this process entails defining, gathering, training, assessing, and applying the deep learning model to complete a particular job or problem."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "R",
   "language": "R",
   "name": "ir"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "4.0.5"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 47.245449,
   "end_time": "2023-05-29T16:54:09.992530",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-05-29T16:53:22.747081",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
