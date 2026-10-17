import React from 'react';

export default function ComponentInspector({ components = [] }) {
  return (
    <div className="component-grid">
      {components.map((c) => (
        <div key={c.id} className="component-item">
          <span className="comp-num">#{c.id}</span>
          <div>
            <strong>{c.name}</strong>
            <small>{c.file}</small>
          </div>
          <span className="comp-status">{c.status}</span>
        </div>
      ))}
    </div>
  );
}
