/** Predict phase completion after recording the current lesson, without mutating progress. */
export function phaseCompletion(active: number, completed: readonly string[], lessonIds: readonly string[], phaseIndices: readonly number[]) {
  const validPhase = phaseIndices.length > 0 && phaseIndices.includes(active)
    && phaseIndices.every(index => Number.isInteger(index) && index >= 0 && index < lessonIds.length);
  if (!validPhase) return { ready: false, nextUnfinished: undefined };
  const nextUnfinished = phaseIndices.find(index => index !== active && !completed.includes(lessonIds[index]));
  return { ready: nextUnfinished === undefined, nextUnfinished };
}
