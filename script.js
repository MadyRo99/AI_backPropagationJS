// A simple backpropagation JS AI making predictions based on the given training data set.

// training data set
var dataB1 = [1,   1,   0];
var dataB2 = [2,   1,   0];
var dataB3 = [2, 0.5, 0];
var dataB4 = [3,   1, 0];

var dataR1 = [3,   1.5, 1];
var dataR2 = [3.5, 0.5, 1];
var dataR3 = [4,   1.5, 1];
var dataR4 = [5.5,   1, 1];

// unknown type (data we want to find)
var dataU = [2,  1, "It should be 1 - Blue"];

var all_points = [dataB1, dataB2, dataB3, dataB4, dataR1, dataR2, dataR3, dataR4];

function sigmoid(x) {
  return 1/(1 + Math.exp(-x));
}

// training
function train() {
  let w1 = Math.random() * 0.2 - 0.1;
  let w2 = Math.random() * 0.2- 0.1;
  let b = Math.random() * 0.2 - 0.1;
  let learning_rate = 0.2;
  for (let iter = 0; iter < 50000; iter++) {

    let random_idx = Math.floor(Math.random() * all_points.length);
    let point = all_points[random_idx];
    let target = point[2];

    let z = w1 * point[0] + w2 * point[1] + b;
    let pred = sigmoid(z);

    let cost = (pred - target) ** 2;

    let dcost_dpred = 2 * (pred - target);

    let dpred_dz = sigmoid(z) * (1 - sigmoid(z));

    let dz_dw1 = point[0];
    let dz_dw2 = point[1];
    let dz_db = 1;

    let dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1;
    let dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2;
    let dcost_db =  dcost_dpred * dpred_dz * dz_db;

    w1 -= learning_rate * dcost_dw1;
    w2 -= learning_rate * dcost_dw2;
    b -= learning_rate * dcost_db;
  }

  return {w1: w1, w2: w2, b: b};
}

var date = train();
var ready = date.w1 * dataU[0] + date.w2 * dataU[1] + date.b;
var prediction = sigmoid(ready);
var message;

// display the result
if (prediction > 0.5) message = "The flower is red.";
    else message = "The flower is blue.";
alert (message);