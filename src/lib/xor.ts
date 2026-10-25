export type Parameters = {
  w1: number[][];
  b1: number[];
  w2: number[];
  b2: number;
};

export type XorSnapshot = {
  parameters: Parameters;
  iteration: number;
  loss: number;
  accuracy: number;
  samples: {
    inputs: [number, number];
    target: number;
    prediction: number;
    probability: number;
  }[];
  selected: {
    index: number;
    inputs: [number, number];
    target: number;
    z1: number[];
    a1: number[];
    z2: number;
    output: number;
    loss: number;
  };
  gradients: Parameters;
  sample_gradients: Parameters;
  grid: {
    size: number;
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
    probabilities: number[];
  };
};

export type ParameterUpdate = {
  name: string;
  before: number;
  gradient: number;
  after: number;
};
export type HistoryPoint = { step: number; loss: number; accuracy: number };
export type XorStepResponse = {
  snapshot: XorSnapshot;
  history: HistoryPoint[];
  updates: ParameterUpdate[];
};

export function flattenParameters(
  parameters: Parameters,
): { name: string; value: number }[] {
  return [
    ...parameters.w1.flatMap((row, i) =>
      row.map((value, j) => ({ name: `w1[${i},${j}]`, value })),
    ),
    ...parameters.b1.map((value, j) => ({ name: `b1[${j}]`, value })),
    ...parameters.w2.map((value, j) => ({ name: `w2[${j}]`, value })),
    { name: "b2", value: parameters.b2 },
  ];
}
