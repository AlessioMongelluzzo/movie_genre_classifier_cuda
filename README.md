# NLP Movie Genre Classifier
Inference wrapper for NLP model performing inference with CUDA (if available) on movies' genres from title and textual description (english).
Training performed on [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv).

## Requirements
- [Docker](https://docs.docker.com/get-docker/)
- [Nvidia CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
- [Nvidia cuDNN 8](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Usage
clone repository and cd to code directory
```bash
git clone https://github.com/AlessioMongelluzzo/movie_genre_classifier.git
cd movie_genre_classifier/code/
```
build the container
```bash
sudo docker build -t genclf .
```
run the container and publish port 5000 (model app runs on port 5000)
```bash
sudo docker run --privileged --gpus all -d -p 5000:5000 genclf
```
## Testing
The model runs on http://localhost:5000/, you can test inference by performing a POST API request to http://localhost:5000/predict with a JSON payload like
```json
{
    "title": "insert title here",
    "description": "insert description here"
}
```
e.g., with curl
```bash
curl --location --request POST 'http://localhost:5000/predict' --header 'Content-Type: application/json' --data-raw '{"title": "insert your title here", "description": "insert description here"}'
```
example:
```bash
curl --location --request POST 'http://localhost:5000/predict' --header 'Content-Type: application/json' --data-raw '{"title": "The Super Pickle Model", "description": "An amazing Pickle model with super powers fights every machine learning problem!"}'
{"title": "The Super Pickle Model", "description": "An amazing Pickle model with super powers fights every machine learning problem!", "genre": "Animation, Comedy, Family, Adventure"}
 ```
