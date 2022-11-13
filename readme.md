## 1. Download the model

We're using this [mode](https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_large_100_224/classification/5/default/1).

```shell
./prepare.sh
```

By running this script, you'll get a folder called `model`.

## 2. Run

Before starting, you must have a folder called `./download/` which contains JPEG files.

```
npm start
``

After starting the script, you'll get a Sqlite database file called `labels.db`.