{
 "cells": [
  {
   "cell_type": "raw",
   "id": "9cab64cf",
   "metadata": {
    "papermill": {
     "duration": 0.005311,
     "end_time": "2023-05-29T16:33:05.454820",
     "exception": false,
     "start_time": "2023-05-29T16:33:05.449509",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Load required libraries\n",
    "library(keras)\n",
    "library(tensorflow)\n",
    "library(ggplot2)\n",
    "\n",
    "# Load MNIST dataset\n",
    "mnist <- dataset_mnist()\n",
    "x_train <- mnist$train$x\n",
    "y_train <- mnist$train$y\n",
    "x_test <- mnist$test$x\n",
    "y_test <- mnist$test$y\n",
    "\n",
    "# Preprocess the data\n",
    "x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))\n",
    "x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))\n",
    "x_train <- x_train / 255\n",
    "x_test <- x_test / 255\n",
    "\n",
    "# Define the model architecture\n",
    "model <- keras_model_sequential()\n",
    "model %>%\n",
    "  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = \"relu\", input_shape = c(28, 28, 1)) %>%\n",
    "  layer_max_pooling_2d(pool_size = c(2, 2)) %>%\n",
    "  layer_dropout(0.25) %>%\n",
    "  layer_flatten() %>%\n",
    "  layer_dense(units = 128, activation = \"relu\") %>%\n",
    "  layer_dropout(0.5) %>%\n",
    "  layer_dense(units = 10, activation = \"softmax\")\n",
    "\n",
    "# Compile the model\n",
    "model %>% compile(\n",
    "  loss = \"sparse_categorical_crossentropy\",\n",
    "  optimizer = optimizer_adam(),\n",
    "  metrics = c(\"accuracy\")\n",
    ")\n",
    "\n",
    "# Train the model\n",
    "history <- model %>% fit(\n",
    "  x_train, y_train,\n",
    "  epochs = 10,\n",
    "  batch_size = 128,\n",
    "  validation_split = 0.2\n",
    ")\n",
    "\n",
    "# Evaluate the model\n",
    "accuracy <- model %>% evaluate(x_test, y_test)[[2]]\n",
    "cat(\"Accuracy: \", accuracy, \"\\n\")\n",
    "\n",
    "# Plot accuracy and loss curves\n",
    "plot(history)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "977f05e4",
   "metadata": {
    "papermill": {
     "duration": 0.004024,
     "end_time": "2023-05-29T16:33:05.463045",
     "exception": false,
     "start_time": "2023-05-29T16:33:05.459021",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Loading Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "98715619",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:05.475751Z",
     "iopub.status.busy": "2023-05-29T16:33:05.473449Z",
     "iopub.status.idle": "2023-05-29T16:33:06.951357Z",
     "shell.execute_reply": "2023-05-29T16:33:06.949482Z"
    },
    "papermill": {
     "duration": 1.48745,
     "end_time": "2023-05-29T16:33:06.954573",
     "exception": false,
     "start_time": "2023-05-29T16:33:05.467123",
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
   "id": "8c8cb921",
   "metadata": {
    "papermill": {
     "duration": 0.004691,
     "end_time": "2023-05-29T16:33:06.963902",
     "exception": false,
     "start_time": "2023-05-29T16:33:06.959211",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Loading the Data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9534739e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:07.021673Z",
     "iopub.status.busy": "2023-05-29T16:33:06.974076Z",
     "iopub.status.idle": "2023-05-29T16:33:18.360660Z",
     "shell.execute_reply": "2023-05-29T16:33:18.358968Z"
    },
    "papermill": {
     "duration": 11.395544,
     "end_time": "2023-05-29T16:33:18.363788",
     "exception": false,
     "start_time": "2023-05-29T16:33:06.968244",
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
   "id": "f0af8657",
   "metadata": {
    "papermill": {
     "duration": 0.004164,
     "end_time": "2023-05-29T16:33:18.372347",
     "exception": false,
     "start_time": "2023-05-29T16:33:18.368183",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6d3662bc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:18.383418Z",
     "iopub.status.busy": "2023-05-29T16:33:18.382096Z",
     "iopub.status.idle": "2023-05-29T16:33:19.857837Z",
     "shell.execute_reply": "2023-05-29T16:33:19.855943Z"
    },
    "papermill": {
     "duration": 1.484109,
     "end_time": "2023-05-29T16:33:19.860609",
     "exception": false,
     "start_time": "2023-05-29T16:33:18.376500",
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
   "id": "01fb066d",
   "metadata": {
    "papermill": {
     "duration": 0.004148,
     "end_time": "2023-05-29T16:33:19.869108",
     "exception": false,
     "start_time": "2023-05-29T16:33:19.864960",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Defining model Architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e453d9d6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:19.881238Z",
     "iopub.status.busy": "2023-05-29T16:33:19.879947Z",
     "iopub.status.idle": "2023-05-29T16:33:23.481599Z",
     "shell.execute_reply": "2023-05-29T16:33:23.479863Z"
    },
    "papermill": {
     "duration": 3.611343,
     "end_time": "2023-05-29T16:33:23.484539",
     "exception": false,
     "start_time": "2023-05-29T16:33:19.873196",
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
   "id": "774094cb",
   "metadata": {
    "papermill": {
     "duration": 0.004707,
     "end_time": "2023-05-29T16:33:23.493728",
     "exception": false,
     "start_time": "2023-05-29T16:33:23.489021",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Compiling the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2f8c2370",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:23.504784Z",
     "iopub.status.busy": "2023-05-29T16:33:23.503473Z",
     "iopub.status.idle": "2023-05-29T16:33:23.531016Z",
     "shell.execute_reply": "2023-05-29T16:33:23.529570Z"
    },
    "papermill": {
     "duration": 0.03563,
     "end_time": "2023-05-29T16:33:23.533525",
     "exception": false,
     "start_time": "2023-05-29T16:33:23.497895",
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
   "id": "0b193051",
   "metadata": {
    "papermill": {
     "duration": 0.004238,
     "end_time": "2023-05-29T16:33:23.541963",
     "exception": false,
     "start_time": "2023-05-29T16:33:23.537725",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Training the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c9a4607f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:23.553677Z",
     "iopub.status.busy": "2023-05-29T16:33:23.552310Z",
     "iopub.status.idle": "2023-05-29T16:33:49.951017Z",
     "shell.execute_reply": "2023-05-29T16:33:49.949175Z"
    },
    "papermill": {
     "duration": 26.408318,
     "end_time": "2023-05-29T16:33:49.954464",
     "exception": false,
     "start_time": "2023-05-29T16:33:23.546146",
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
   "id": "a1563ddb",
   "metadata": {
    "papermill": {
     "duration": 0.004375,
     "end_time": "2023-05-29T16:33:49.964400",
     "exception": false,
     "start_time": "2023-05-29T16:33:49.960025",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Evaluating the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "45aa4bfd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:49.975618Z",
     "iopub.status.busy": "2023-05-29T16:33:49.974276Z",
     "iopub.status.idle": "2023-05-29T16:33:50.961751Z",
     "shell.execute_reply": "2023-05-29T16:33:50.958728Z"
    },
    "papermill": {
     "duration": 0.996133,
     "end_time": "2023-05-29T16:33:50.964686",
     "exception": false,
     "start_time": "2023-05-29T16:33:49.968553",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.9882 \n"
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
   "id": "bc662d88",
   "metadata": {
    "papermill": {
     "duration": 0.00508,
     "end_time": "2023-05-29T16:33:50.974994",
     "exception": false,
     "start_time": "2023-05-29T16:33:50.969914",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Printing the History"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fb07d3d9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:33:50.987069Z",
     "iopub.status.busy": "2023-05-29T16:33:50.985720Z",
     "iopub.status.idle": "2023-05-29T16:33:51.002604Z",
     "shell.execute_reply": "2023-05-29T16:33:51.000975Z"
    },
    "papermill": {
     "duration": 0.025519,
     "end_time": "2023-05-29T16:33:51.005055",
     "exception": false,
     "start_time": "2023-05-29T16:33:50.979536",
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
      "        loss: 0.04632\n",
      "    accuracy: 0.9853\n",
      "    val_loss: 0.04445\n",
      "val_accuracy: 0.9877 \n"
     ]
    }
   ],
   "source": [
    "print(history)"
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
   "duration": 50.012443,
   "end_time": "2023-05-29T16:33:52.332699",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-05-29T16:33:02.320256",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
