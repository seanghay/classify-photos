import path from 'path'
import fs from 'fs/promises'
import fg from 'fast-glob'
import tf from '@tensorflow/tfjs-node'
import PQueue from 'p-queue'
import fsSync from 'fs';

let stats = 0;

await fs.mkdir('labels', { recursive: true })

function preprocess(imageTensor) {
  const widthToHeight = imageTensor.shape[1] / imageTensor.shape[0];
  let squareCrop;

  if (widthToHeight > 1) {
    const heightToWidth = imageTensor.shape[0] / imageTensor.shape[1];
    const cropTop = (1 - heightToWidth) / 2;
    const cropBottom = 1 - cropTop;
    squareCrop = [[cropTop, 0, cropBottom, 1]];
  } else {
    const cropLeft = (1 - widthToHeight) / 2;
    const cropRight = 1 - cropLeft;
    squareCrop = [[0, cropLeft, 1, cropRight]];
  }
  const crop = tf.image.cropAndResize(
    tf.expandDims(imageTensor), squareCrop, [0], [224, 224]);
  return crop.div(255);
}

function printMem() {
  const heap = process.memoryUsage().heapUsed / 1024 / 1024;
  const heapInMB = (Math.round(heap * 100) / 100) + "MB";
  console.log(`Memory usage: ${heapInMB}, status: ${stats}`);
}

const modelPath = new URL('./model/model.json', import.meta.url).href;
const model = await tf.loadGraphModel(modelPath);
const queue = new PQueue({ concurrency: 10 });

for await (const file of fg.stream('./downloads/*.jpg')) {
  
  queue.add(async () => {
    const filename = path.parse(file).name + ".json";
    const outputPath = path.join("labels", filename);
    
    if (fsSync.existsSync(outputPath)) {
      stats++;
      return;
    }    

    printMem();
    const buffer = await fs.readFile(file);
    const imageTensor = tf.node.decodeJpeg(buffer, 3);
    const predictions = model.predict(preprocess(imageTensor))
    const { classNames } = model.metadata;
    const values = await predictions.squeeze().data();
    const labels = [...values]
      .sort((a, b) => b - a).slice(0, 10)
      .map(i => classNames[values.indexOf(i)])

    const json = JSON.stringify(labels)
    await fs.writeFile(outputPath, json, 'utf8');
    stats++;
  })

  await queue.onSizeLessThan(queue.concurrency);
}
