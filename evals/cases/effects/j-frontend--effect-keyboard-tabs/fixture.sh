set -euo pipefail
mkdir -p src
cat > src/Tabs.tsx <<'TSX'
import { useState, type ReactNode } from "react";

export type Tab = { id: string; label: string; content: ReactNode };

export function Tabs({ tabs }: { tabs: Tab[] }) {
  const [active, setActive] = useState(tabs[0]?.id);

  return (
    <div className="tabs">
      <div className="tab-strip">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className={tab.id === active ? "tab tab--active" : "tab"}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </div>
        ))}
      </div>
      {tabs.map((tab) => (
        <div key={tab.id} className="panel" hidden={tab.id !== active}>
          {tab.content}
        </div>
      ))}
    </div>
  );
}
TSX
cat > src/AccountPage.tsx <<'TSX'
import { Tabs } from "./Tabs";

export function AccountPage() {
  return (
    <Tabs
      tabs={[
        { id: "profile", label: "Profile", content: <p>Name and photo</p> },
        { id: "billing", label: "Billing", content: <p>Cards and invoices</p> },
        { id: "security", label: "Security", content: <p>Password and sessions</p> },
      ]}
    />
  );
}
TSX
