import path from 'path'
import fs from 'fs/promises'
import fg from 'fast-glob'
import tf from '@tensorflow/tfjs-node'
import PQueue from 'p-queue'
import fsSync from 'fs';
import Knex from 'knex'

const knex = Knex({
  client: "better-sqlite3",
  connection: {
    filename: "./labels.db"
  },
  useNullAsDefault: true,
})

if (!await knex.schema.hasTable("labels")) {
  await knex.schema.createTable('labels', t => {
    t.increments().primary();
    t.string('label').notNullable();
    t.string('filename').notNullable();
    t.unique(['label', 'filename']);
  })
}

const Label = () => knex("labels");
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
  const expand = tf.expandDims(imageTensor)
  const crop = tf.image.cropAndResize(expand, squareCrop, [0], [224, 224]);
  const r = crop.div(255)
  return [r, () => tf.dispose([squareCrop, r, crop, expand])];
}

function printMem() {
  const heap = process.memoryUsage().rss / 1024 / 1024;
  const heapInMB = (Math.round(heap * 100) / 100) + "MB";
  console.log(`Memory usage: ${heapInMB}, status: ${stats}`);
}

const modelPath = new URL('./model/model.json', import.meta.url).href;
const model = await tf.loadGraphModel(modelPath);
const queue = new PQueue({ concurrency: 20 });

for await (const file of fg.stream('./downloads/*.jpg')) {

  queue.add(async () => {
    printMem();
    const filename = path.parse(file).name + ".json";

    const exists = await Label().select('id').where('filename', filename).first()
    if (exists) {
      stats++;
      return;
    }

    const outputPath = path.join("labels", filename);
    const buffer = await fs.readFile(file);
    const imageTensor = tf.node.decodeJpeg(buffer, 3);
    const [prep, dispose] = preprocess(imageTensor)
    const predictions = model.predict(prep);
    const { classNames } = model.metadata;
    const values = await predictions.squeeze().data();
    const labels = [...values]
      .sort((a, b) => b - a).slice(0, 10)
      .map(i => classNames[values.indexOf(i)])

    tf.dispose(predictions)
    dispose()
    tf.dispose(imageTensor)

    // store labels
    console.time('insert: ' + filename)
    await Promise.all(labels.map(label => Label().insert({
      label,
      filename
    }).onConflict().ignore()))

    console.timeEnd('insert: ' + filename)
    stats++;
  })

  await queue.onSizeLessThan(queue.concurrency);
}

await queue.onIdle();
model.dispose();
await knex.destroy();
