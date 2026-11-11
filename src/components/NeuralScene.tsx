import { useId } from "react";
import "./neural-scene.css";

type Node = { x: number; y: number };

const layers: Node[][] = [
  [89, 119, 149, 179].map((y, i) => ({ x: 66 + i * 2, y })),
  [54, 86, 118, 150, 182, 214].map((y, i) => ({ x: 165 + i * 2, y })),
  [66, 101, 136, 171, 206].map((y, i) => ({ x: 272 + i * 2, y })),
  [103, 137, 171].map((y, i) => ({ x: 389 + i * 2, y })),
];

function connect(from: Node, to: Node) {
  const bend = (to.x - from.x) * 0.38;
  return `M ${from.x} ${from.y} C ${from.x + bend} ${from.y}, ${to.x - bend} ${to.y}, ${to.x} ${to.y}`;
}

/** Decorative topology: the illustration does not represent a trained model. */
export default function NeuralScene({
  phase = 1,
  effects = true,
}: {
  phase?: number;
  effects?: boolean;
}) {
  const id = useId().replace(/:/g, "");
  const paths = layers.slice(0, -1).flatMap((layer, layerIndex) =>
    layer.flatMap((from, sourceIndex) =>
      layers[layerIndex + 1].map((to, targetIndex) => ({
        d: connect(from, to),
        key: `${layerIndex}-${sourceIndex}-${targetIndex}`,
        signal: (sourceIndex + targetIndex + phase) % 5 === 0,
        delay: -((sourceIndex * 2 + targetIndex + layerIndex) % 9),
      })),
    ),
  );

  return (
    <div
      className={`neural-scene${effects ? "" : " neural-scene-paused"}`}
      aria-hidden="true"
    >
      <svg viewBox="0 0 460 280" fill="none" focusable="false">
        <defs>
          <radialGradient id={`${id}-aura`}>
            <stop stopColor="var(--fwd)" stopOpacity=".15" />
            <stop offset="1" stopColor="var(--fwd)" stopOpacity="0" />
          </radialGradient>
          <linearGradient id={`${id}-link`} x1="60" y1="130" x2="400" y2="130" gradientUnits="userSpaceOnUse">
            <stop stopColor="var(--fwd)" stopOpacity=".13" />
            <stop offset=".5" stopColor="var(--fwd)" stopOpacity=".3" />
            <stop offset="1" stopColor="var(--fwd)" stopOpacity=".26" />
          </linearGradient>
          <linearGradient id={`${id}-plane`} x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="var(--fwd)" stopOpacity=".07" />
            <stop offset="1" stopColor="var(--fwd)" stopOpacity=".01" />
          </linearGradient>
          <filter id={`${id}-glow`} x="-150%" y="-150%" width="400%" height="400%">
            <feGaussianBlur stdDeviation="2.6" />
          </filter>
          <pattern id={`${id}-dots`} width="18" height="18" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r=".65" fill="var(--fwd)" opacity=".18" />
          </pattern>
        </defs>

        <rect x="12" y="17" width="436" height="246" fill={`url(#${id}-dots)`} />
        <ellipse cx="238" cy="139" rx="210" ry="135" fill={`url(#${id}-aura)`} />

        <g className="neural-orbits" stroke="var(--fwd)" strokeWidth=".65">
          <ellipse cx="233" cy="139" rx="203" ry="84" transform="rotate(-15 233 139)" opacity=".14" />
          <ellipse cx="233" cy="139" rx="186" ry="99" transform="rotate(15 233 139)" opacity=".09" />
          <ellipse cx="233" cy="139" rx="210" ry="70" transform="rotate(-15 233 139)" opacity=".09" strokeDasharray="2 9" />
        </g>

        <g className="neural-coordinate-lines" stroke="var(--fwd)" strokeWidth=".6" opacity=".13">
          <path d="M29 239H431 M29 239L55 218 M431 239L406 218 M233 249V255" />
          {[66, 171, 277, 391].map((x) => <path key={x} d={`M${x} 236v6`} />)}
          <path d="M14 49V27H36 M424 27H446V49 M14 229V251H36 M424 251H446V229" opacity=".9" />
        </g>

        <g fill={`url(#${id}-plane)`} stroke="var(--fwd)" strokeWidth=".7">
          <path d="M50 72L84 84L96 206L62 194Z" strokeOpacity=".19" />
          <path d="M143 30L181 44L196 238L158 224Z" strokeOpacity=".25" />
          <path d="M252 44L290 58L303 232L265 218Z" strokeOpacity=".23" />
          <path d="M372 87L404 97L414 198L382 188Z" strokeOpacity=".22" />
        </g>

        <g stroke={`url(#${id}-link)`} strokeWidth=".8">
          {paths.map((path) => <path key={path.key} d={path.d} />)}
        </g>
        <g className="neural-signals" stroke="var(--fwd)" strokeWidth="1.2" strokeLinecap="round">
          {paths.filter((path) => path.signal).map((path) => (
            <path
              key={path.key}
              d={path.d}
              pathLength="100"
              style={{ animationDelay: `${path.delay}s` }}
            />
          ))}
        </g>

        {layers.map((layer, layerIndex) => (
          <g key={layerIndex} className={`neural-node-layer neural-node-layer-${layerIndex}`}>
            {layer.map((node, nodeIndex) => {
              const color = layerIndex === 3 ? "var(--output-node)" : "var(--fwd)";
              const highlighted = (nodeIndex + layerIndex + phase) % 3 === 0;
              return (
                <g key={nodeIndex} className={highlighted ? "neural-node-lit" : undefined} style={{ animationDelay: `${-nodeIndex * 0.7}s` }}>
                  <circle cx={node.x} cy={node.y} r={highlighted ? 9 : 6} fill={color} opacity={highlighted ? 0.28 : 0.12} filter={`url(#${id}-glow)`} />
                  <circle cx={node.x} cy={node.y} r={highlighted ? 5.1 : 4} fill="var(--surface-1)" stroke={color} strokeOpacity={highlighted ? 1 : 0.7} strokeWidth="1.1" />
                  <circle cx={node.x} cy={node.y} r={highlighted ? 2.1 : 1.25} fill={color} opacity={highlighted ? 1 : 0.65} />
                </g>
              );
            })}
          </g>
        ))}

        <g className="neural-orbit-particles">
          <circle cx="419" cy="98" r="2.1" fill="var(--fwd)" />
          <circle cx="46" cy="181" r="1.5" fill="var(--fwd)" />
          <circle cx="316" cy="43" r="1.1" fill="var(--fwd)" />
          <path d="M320 43h15m-11-4v8" stroke="var(--fwd)" strokeOpacity=".3" strokeWidth=".6" />
        </g>

        <g className="neural-labels" fill="var(--text-3)" fontSize="7.5" fontFamily="'Cascadia Code', Consolas, monospace" letterSpacing="1.6">
          <text x="51" y="261">INPUT</text>
          <text x="184" y="261">HIDDEN LAYERS</text>
          <text x="368" y="261">OUTPUT</text>
          <text x="30" y="24" fontSize="6.5" fill="var(--fwd)">NEURAL TOPOLOGY</text>
          <text x="346" y="24" fontSize="6.5" fill="var(--fwd)">SIGNAL FLOW →</text>
        </g>
      </svg>
    </div>
  );
}
