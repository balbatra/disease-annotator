# Documentation of DiseaseBert

## download the model
Extract the model in the same directory of the Dockerfile please do not change the name of the directory you have unzipped.

https://drive.google.com/file/d/1WlWx0KX0JtM5e9cccm4GmsFa6y2Hq2O5/view?usp=sharing


## Build the docker image
docker build . --tag disease-bert

## Run the server
docker run -p 8080:8080 -d --name disease-bert disease-bert

## Using the API
http://localhost:8080/annotate-diseases?text=blablabla