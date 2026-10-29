import { Component } from "react";
import type { ReactNode } from "react";
import "./experiment-loading.css";

/** A failed on-demand chunk must offer recovery instead of an endless spinner. */
export default class ExperimentBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) return <div className="experiment-load-error" role="alert">
      <strong>实验暂时无法加载</strong>
      <p>请检查连接后重新加载页面。已保存的学习进度会保留。</p>
      <button type="button" className="secondary-button" onClick={() => window.location.reload()}>重新加载页面</button>
    </div>;
    return this.props.children;
  }
}
