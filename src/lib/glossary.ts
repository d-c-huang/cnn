import { lessonDeepDives } from "./lessonDeepDives";

export const glossary = Object.entries(lessonDeepDives).flatMap(([lessonId, content]) => content.terms.map(item => ({ ...item, lessonId })));
export function glossaryId(term: string) { return `term-${encodeURIComponent(term)}`; }
