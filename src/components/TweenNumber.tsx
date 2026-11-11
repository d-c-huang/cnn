import { useTween } from "../lib/useTween";

export default function TweenNumber({ value, digits = 4, suffix = "" }: { value: number; digits?: number; suffix?: string }) {
  const displayed = useTween(value);
  return <span className="tween-number" aria-label={`${value.toFixed(digits)}${suffix}`}><span aria-hidden="true">{displayed.toFixed(digits)}{suffix}</span></span>;
}
