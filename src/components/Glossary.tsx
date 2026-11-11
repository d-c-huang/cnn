import { Fragment, useEffect, useState } from "react";
import { glossary, glossaryId } from "../lib/glossary";

export function GlossaryText({ text }: { text: string }) {
  const terms = [...new Set(glossary.map(item => item.term))].sort((a, b) => b.length - a.length);
  const pattern = new RegExp(`(${terms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g");
  return <>{text.split(pattern).map((part, index) => terms.includes(part) ? <a key={index} className="glossary-link" href={`#${glossaryId(part)}`} onClick={event => {
    event.preventDefault();
    window.dispatchEvent(new CustomEvent("zhiwei-glossary-open", { detail: part }));
  }}>{part}</a> : <Fragment key={index}>{part}</Fragment>)}</>;
}

export default function Glossary() {
  const [order, setOrder] = useState("course");
  const [search, setSearch] = useState("");
  useEffect(() => {
    let frame = 0;
    const open = (event: Event) => {
      setSearch("");
      const panel = document.getElementById("course-glossary") as HTMLDetailsElement | null;
      if (panel) panel.open = true;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const target = document.getElementById(glossaryId((event as CustomEvent<string>).detail));
        target?.focus({ preventScroll: true }); target?.scrollIntoView({ block: "center" });
      });
    };
    window.addEventListener("zhiwei-glossary-open", open);
    return () => { window.removeEventListener("zhiwei-glossary-open", open); cancelAnimationFrame(frame); };
  }, []);
  const items = (order === "alpha" ? [...glossary].sort((a, b) => a.term.localeCompare(b.term, "zh-Hans-CN-u-co-pinyin")) : glossary).filter(item => `${item.term}${item.definition}`.toLowerCase().includes(search.toLowerCase()));
  return <details className="course-glossary" id="course-glossary"><summary>术语索引 · {glossary.length} 个概念</summary><div className="glossary-controls"><label>查找术语<input type="search" value={search} onChange={event => setSearch(event.target.value)} /></label><label>排序<select value={order} onChange={event => setOrder(event.target.value)}><option value="course">课程出现顺序</option><option value="alpha">拼音 / 字母顺序</option></select></label></div><dl>{items.map(item => <div key={`${item.lessonId}-${item.term}`} id={glossaryId(item.term)} tabIndex={-1}><dt>{item.term}<a href={`#${item.lessonId}`}>回到实验</a></dt><dd>{item.definition}</dd></div>)}</dl>{items.length === 0 && <p>没有匹配的术语。</p>}</details>;
}
