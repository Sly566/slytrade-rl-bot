from __future__ import annotations

from dataclasses import dataclass

from slytrade.backtest.execution import ExecutionConfig, Quote, TickExecutionSimulator
from slytrade.backtest.portfolio import Fill, PortfolioState
from slytrade.execution.ledger import TradeLedger
from slytrade.execution.models import ExecutionReport, OrderIntent, OrderStatus
from slytrade.execution.oms import OrderManagementSystem
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails


@dataclass(frozen=True)
class PaperBrokerResult:
    report: ExecutionReport
    equity: float
    approved: bool
    reason: str


class PaperBroker:
    """Safe broker simulator that routes every order through risk and OMS.

    This is the first environment where the production execution path exists:

    OrderIntent -> Guardrails -> OMS -> ExecutionSimulator -> Portfolio -> Ledger
    """

    def __init__(
        self,
        *,
        initial_balance: float = 100_000.0,
        execution_config: ExecutionConfig | None = None,
        guardrail_config: GuardrailConfig | None = None,
        oms: OrderManagementSystem | None = None,
        ledger: TradeLedger | None = None,
    ):
        self.portfolio = PortfolioState(initial_balance=initial_balance)
        self.execution = TickExecutionSimulator(execution_config or ExecutionConfig())
        self.guardrails = TradingGuardrails(guardrail_config or GuardrailConfig(), initial_equity=initial_balance)
        self.oms = oms or OrderManagementSystem()
        self.ledger = ledger or TradeLedger()
        self.last_marks: dict[str, float] = {}

    def submit_order(self, intent: OrderIntent, quote: Quote) -> PaperBrokerResult:
        self.last_marks[quote.symbol] = quote.mid
        equity = self.portfolio.mark_to_market(self.last_marks)
        decision = self.guardrails.approve_order(
            intent,
            equity=equity,
            spread_points=quote.spread / max(self.execution.config.point_size, 1e-12),
            live=False,
        )
        if not decision.approved:
            self.oms.create_order(intent)
            report = ExecutionReport(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                message=decision.reason,
                event_time=quote.time,
            )
            self.oms.apply_report(report)
            return PaperBrokerResult(report, equity, False, decision.reason)

        self.oms.create_order(intent)
        simulated = self.execution.execute(intent, quote)
        self.oms.apply_report(simulated.report)
        if simulated.report.status == OrderStatus.FILLED and simulated.report.avg_fill_price is not None:
            realized = self.portfolio.apply_fill(
                Fill(
                    symbol=intent.symbol,
                    side=intent.side,
                    volume=simulated.report.filled_volume,
                    price=simulated.report.avg_fill_price,
                    commission=simulated.commission,
                    point_value=simulated.point_value,
                )
            )
            self.ledger.record_fill(
                intent,
                volume=simulated.report.filled_volume,
                price=simulated.report.avg_fill_price,
                commission=simulated.commission,
                realized_pnl=realized,
                event_time=simulated.report.event_time,
            )

        equity = self.portfolio.mark_to_market(self.last_marks)
        return PaperBrokerResult(simulated.report, equity, True, simulated.report.message)
