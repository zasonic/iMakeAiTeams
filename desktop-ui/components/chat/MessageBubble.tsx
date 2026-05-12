// desktop-ui/components/chat/MessageBubble.tsx
//
// Layer C2 first slice: extract the leaf message-row renderer from
// ChatView.tsx. MessageBubble is the lowest-coupling component in the
// chat surface — it takes one prop bag and renders one rounded bubble
// with the role-appropriate styling and an optional model/cost footer.
// No state, no callbacks, no imperative refs.
//
// Larger extractions (MessageList wrapping VariableSizeList, MessageInput
// with the slash-menu + paste + drag-drop coupling) are still inside
// ChatView and deferred — see the C2 commit message for the planned order.

import { MessageRenderer } from "@/components/MessageRenderer";

export interface MessageRow {
  id:           string;
  role:         "user" | "assistant" | "system";
  content:      string;
  model_used?:  string;
  cost_usd?:    number;
}

interface MessageBubbleProps {
  msg:                 MessageRow;
  voiceOutputEnabled:  boolean;
}

export function MessageBubble({ msg, voiceOutputEnabled }: MessageBubbleProps) {
  return (
    <div
      className={`max-w-[80%] rounded-xl px-4 py-2 text-sm ${
        msg.role === "user"
          ? "ml-auto bg-accent/15 text-ink border border-accent/20"
          : "bg-bg-2 text-ink border border-line"
      }`}
    >
      <MessageRenderer
        content={msg.content}
        role={msg.role}
        voiceOutputEnabled={voiceOutputEnabled}
      />
      {msg.model_used && (
        <div className="text-[11px] text-ink-faint mt-2">
          {msg.model_used}
          {typeof msg.cost_usd === "number" && ` · $${msg.cost_usd.toFixed(4)}`}
        </div>
      )}
    </div>
  );
}
