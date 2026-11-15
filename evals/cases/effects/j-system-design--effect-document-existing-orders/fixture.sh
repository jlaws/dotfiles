set -euo pipefail
mkdir -p src
cat > src/orders.js <<'JS'
import { reserve } from "./inventory.js";
import { chargeCard, sendReceipt, auditLog } from "./services.js";

export function placeOrder(order) {
  reserve(order.items);
  chargeCard(order.card, order.total);
  sendReceipt(order.email, order);
  auditLog("order placed", order);
  return { id: order.id, status: "placed" };
}

export function orderStatus(id) {
  return { id, status: "placed" };
}
JS
cat > src/inventory.js <<'JS'
import { orderStatus } from "./orders.js";

const stock = new Map();

export function reserve(items) {
  for (const item of items) {
    stock.set(item.sku, (stock.get(item.sku) || 0) - item.qty);
  }
}

export function restockIfCancelled(id, items) {
  if (orderStatus(id).status === "cancelled") {
    for (const item of items) stock.set(item.sku, (stock.get(item.sku) || 0) + item.qty);
  }
}
JS
cat > src/services.js <<'JS'
export function chargeCard(card, total) {
  return { ok: true, card, total };
}

export function sendReceipt(email, order) {
  return { to: email, order };
}

export function auditLog(message, payload) {
  console.log(message, JSON.stringify(payload));
}

export function formatMoney(cents) {
  return (cents / 100).toFixed(2);
}

export function validateCard(card) {
  return Boolean(card);
}

export function renderEmailTemplate(order) {
  return `Thanks for order ${order.id}`;
}
JS
