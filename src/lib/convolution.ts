export type Matrix = number[][];

function checkMatrix(matrix: Matrix, name: string, allowEmpty = false) {
  if (allowEmpty && matrix.length === 0) return;
  const width = matrix[0]?.length ?? 0;
  if (!matrix.length || !width || matrix.some(row => row.length !== width || row.some(value => !Number.isFinite(value)))) {
    throw new Error(`${name} must be a non-empty rectangular matrix of finite numbers`);
  }
}

function checkGeometry(kernelSize: number, stride: number, padding: number) {
  if (!Number.isInteger(kernelSize) || kernelSize < 1 || !Number.isInteger(stride) || stride < 1 || !Number.isInteger(padding) || padding < 0) {
    throw new Error("Kernel size and stride must be positive integers; padding must be a non-negative integer");
  }
}

export function outputSize(inputSize: number, kernelSize: number, stride = 1, padding = 0): number {
  checkGeometry(kernelSize, stride, padding);
  if (!Number.isInteger(inputSize) || inputSize < 0) throw new Error("Input size must be a non-negative integer");
  return Math.max(0, Math.floor((inputSize + 2 * padding - kernelSize) / stride) + 1);
}

/** Deep-learning convolution uses cross-correlation: weights are not flipped. */
export function crossCorrelate(input: Matrix, kernel: Matrix, stride = 1, padding = 0): Matrix {
  checkMatrix(input, "Input");
  checkMatrix(kernel, "Kernel");
  checkGeometry(kernel.length, stride, padding);
  const rows = outputSize(input.length, kernel.length, stride, padding);
  const columns = outputSize(input[0].length, kernel[0].length, stride, padding);
  if (!rows || !columns) return [];
  return Array.from({ length: rows }, (_, row) => Array.from({ length: columns }, (_, column) => {
    let sum = 0;
    for (let kr = 0; kr < kernel.length; kr++) {
      for (let kc = 0; kc < kernel[0].length; kc++) {
        sum += (input[row * stride + kr - padding]?.[column * stride + kc - padding] ?? 0) * kernel[kr][kc];
      }
    }
    return sum;
  }));
}

export function relu(matrix: Matrix): Matrix {
  checkMatrix(matrix, "Input", true);
  return matrix.map(row => row.map(value => Math.max(0, value)));
}

/** Valid pooling drops the incomplete window on the bottom and right edges. */
export function maxPool2d(matrix: Matrix, size = 2, stride = 2): Matrix {
  checkMatrix(matrix, "Input", true);
  checkGeometry(size, stride, 0);
  if (!matrix.length) return [];
  const rows = outputSize(matrix.length, size, stride);
  const columns = outputSize(matrix[0].length, size, stride);
  if (!rows || !columns) return [];
  return Array.from({ length: rows }, (_, row) => Array.from({ length: columns }, (_, column) => {
    let maximum = -Infinity;
    for (let dr = 0; dr < size; dr++) {
      for (let dc = 0; dc < size; dc++) maximum = Math.max(maximum, matrix[row * stride + dr][column * stride + dc]);
    }
    return maximum;
  }));
}

export function convolutionWindow(input: Matrix, kernel: Matrix, outRow: number, outColumn: number, stride = 1, padding = 0) {
  checkMatrix(input, "Input");
  checkMatrix(kernel, "Kernel");
  checkGeometry(kernel.length, stride, padding);
  const rows = outputSize(input.length, kernel.length, stride, padding);
  const columns = outputSize(input[0].length, kernel[0].length, stride, padding);
  if (!Number.isInteger(outRow) || !Number.isInteger(outColumn) || outRow < 0 || outColumn < 0 || outRow >= rows || outColumn >= columns) {
    throw new Error("Selected output coordinate is outside the feature map");
  }
  return kernel.flatMap((values, kr) => values.map((weight, kc) => {
    const row = outRow * stride + kr - padding;
    const column = outColumn * stride + kc - padding;
    const padded = row < 0 || column < 0 || row >= input.length || column >= input[0].length;
    const value = padded ? 0 : input[row][column];
    return { row, column, input: value, kernel: weight, product: value * weight, padded };
  }));
}
