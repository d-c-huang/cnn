import { useMemo } from "react";
import { renderToString } from "katex";
import "katex/dist/katex.min.css";
import "./math-formula.css";

/** Explicit TeX source keeps mathematical notation separate from explanatory prose. */
export function MathFormula({ tex, block = false, className = "" }: {
  tex: string;
  block?: boolean;
  className?: string;
}) {
  const html = useMemo(() => renderToString(tex, {
    displayMode: block,
    output: "htmlAndMathml",
    throwOnError: false,
    trust: false,
    strict: "ignore",
  }), [tex, block]);

  const Tag = block ? "div" : "span";
  return <Tag className={`math-formula ${block ? "math-block" : "math-inline"} ${className}`}
    dangerouslySetInnerHTML={{ __html: html }} />;
}
