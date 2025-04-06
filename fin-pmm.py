import logging
import os
from decimal import Decimal
from typing import Dict, List
import numpy as np

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class ImprovedPMMConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    exchange: str = Field("binance_paper_trade", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange where the bot will trade"))
    trading_pair: str = Field("DOGE-USDT", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair in which the bot will place orders"))
    order_amount: Decimal = Field(25, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order amount (denominated in base asset)"))
    bid_spread: Decimal = Field(0.001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bid order spread (in percent)"))
    ask_spread: Decimal = Field(0.001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Ask order spread (in percent)"))
    min_spread: Decimal = Field(0.0005, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum spread (in percent)"))
    max_spread: Decimal = Field(0.05, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximum spread (in percent)"))
    order_refresh_time: int = Field(10, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order refresh time (in seconds)"))
    volatility_adjustment: Decimal = Field(0.75, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Volatility adjustment factor (0.1-1.0)"))
    depth_factor: Decimal = Field(0.3, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order book depth factor (0.1-1.0)"))
    stop_loss_pct: Decimal = Field(0.05, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Stop loss percentage"))
    take_profit_pct: Decimal = Field(0.03, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Take profit percentage"))
    price_type: str = Field("mid", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Price type to use (mid or last)"))
    enable_aggressive_mode: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Enable aggressive mode to place orders closer to mid price when market moves (True/False)"))
    sma_period: int = Field(20, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "SMA period"))
    bollinger_window: int = Field(20, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bollinger Bands window"))
    bollinger_stddev: Decimal = Field(2, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bollinger Bands standard deviation"))


class ImprovedPMM(ScriptStrategyBase):
    price_samples = []
    volatility_window = 100
    price_movement_trend = 0
    price_source = PriceType.MidPrice

    @classmethod
    def init_markets(cls, config: ImprovedPMMConfig):
        cls.markets = {config.exchange: {config.trading_pair}}
        cls.price_source = PriceType.LastTrade if config.price_type == "last" else PriceType.MidPrice

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ImprovedPMMConfig):
        super().__init__(connectors)
        self.config = config
        self.create_timestamp = 0
        self.entry_price = None
        self.last_mid_price = None
        self.price_history = []
        self.original_bid_spread = float(self.config.bid_spread)
        self.original_ask_spread = float(self.config.ask_spread)

    def on_tick(self):
        self.logger().debug(f"[Tick] Now: {self.current_timestamp}, Next refresh at: {self.create_timestamp}")

        if self.create_timestamp <= self.current_timestamp:
            current_price = self.connectors[self.config.exchange].get_price_by_type(
                self.config.trading_pair, self.price_source)

            # Store price history for SMA and Bollinger Bands
            self.price_history.append(float(current_price))
            if len(self.price_history) > self.config.bollinger_window:
                self.price_history.pop(0)

            if self.last_mid_price is not None:
                if current_price > self.last_mid_price * Decimal("1.001"):
                    self.price_movement_trend = 1
                elif current_price < self.last_mid_price * Decimal("0.999"):
                    self.price_movement_trend = -1
                else:
                    self.price_movement_trend = 0
                self.logger().info(f"Price trend: {self.price_movement_trend} | Current: {current_price}, Prev: {self.last_mid_price}")

            self.last_mid_price = current_price
            self.price_samples.append(float(current_price))
            if len(self.price_samples) > self.volatility_window:
                self.price_samples.pop(0)

            self.cancel_all_orders()
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)

            if not proposal_adjusted:
                self.logger().warning("No orders placed — proposal rejected by budget checker.")
            else:
                self.place_orders(proposal_adjusted)

            active_orders = self.get_active_orders(self.config.exchange)
            self.logger().debug(f"Active Orders: {[o.client_order_id for o in active_orders]}")

            self.create_timestamp = self.current_timestamp + self.config.order_refresh_time
            self.log_with_clock(logging.INFO, self.format_status())

    def create_proposal(self) -> List[OrderCandidate]:
        ref_price = self.connectors[self.config.exchange].get_price_by_type(
            self.config.trading_pair, self.price_source)

        if len(self.price_samples) > 1:
            mean_price = sum(self.price_samples) / len(self.price_samples)
            variance = sum((p - mean_price) ** 2 for p in self.price_samples) / len(self.price_samples)
            volatility = Decimal(variance ** 0.5 / mean_price)
        else:
            volatility = Decimal("0")

        # Calculate SMA and Bollinger Bands
        if len(self.price_history) >= self.config.bollinger_window:
            sma = float(np.mean(self.price_history))
            stddev = float(np.std(self.price_history))
            upper_band = sma + float(self.config.bollinger_stddev) * stddev
            lower_band = sma - float(self.config.bollinger_stddev) * stddev
            
            # Trend detection via SMA slope
            if len(self.price_history) >= self.config.sma_period:
                short_sma = np.mean(self.price_history[-self.config.sma_period:])
                slope = (float(ref_price) - short_sma) / short_sma
                self.logger().info(f"SMA: {short_sma:.4f}, Slope: {slope:.4f}")
                # Enhance trend detection
                if slope > 0.001:
                    self.price_movement_trend = 1
                elif slope < -0.001:
                    self.price_movement_trend = -1
                
            # Log Bollinger Bands
            self.logger().info(f"Bollinger Bands: [{lower_band:.4f}, {upper_band:.4f}]")
            
            # Adjust spreads based on Bollinger Bands
            bb_adjustment = Decimal("0")
            
            if float(ref_price) > upper_band:
                bb_adjustment = Decimal("0.001")  # Increase spreads when price above upper band
                self.logger().info("Price above upper Bollinger Band - increasing spreads")
            elif float(ref_price) < lower_band:
                bb_adjustment = Decimal("-0.0005")  # Decrease spreads when price below lower band
                self.logger().info("Price below lower Bollinger Band - decreasing spreads")

        order_book = self.connectors[self.config.exchange].get_order_book(self.config.trading_pair)
        depth_threshold = float(ref_price) * 0.01

        bid_depth = sum(float(bid[1]) for bid in order_book.bid_entries()
                        if float(ref_price) - float(bid[0]) <= depth_threshold)
        ask_depth = sum(float(ask[1]) for ask in order_book.ask_entries()
                        if float(ask[0]) - float(ref_price) <= depth_threshold)

        total_depth = bid_depth + ask_depth
        bid_ratio = bid_depth / total_depth if total_depth > 0 else Decimal("0.5")
        ask_ratio = ask_depth / total_depth if total_depth > 0 else Decimal("0.5")

        vol_adj = volatility * self.config.volatility_adjustment
        adj_bid_spread = self.config.bid_spread + vol_adj - Decimal(str(bid_ratio)) * self.config.depth_factor
        adj_ask_spread = self.config.ask_spread + vol_adj - Decimal(str(ask_ratio)) * self.config.depth_factor
        
        # Apply Bollinger Band adjustment if available
        if 'bb_adjustment' in locals():
            adj_bid_spread += bb_adjustment
            adj_ask_spread += bb_adjustment

        if self.config.enable_aggressive_mode:
            aggression = Decimal("0.003")
            if self.price_movement_trend == 1:
                adj_ask_spread -= aggression
                self.logger().info("Uptrend → More aggressive on sell")
            elif self.price_movement_trend == -1:
                adj_bid_spread -= aggression
                self.logger().info("Downtrend → More aggressive on buy")

        adj_bid_spread = max(self.config.min_spread, min(self.config.max_spread, adj_bid_spread))
        adj_ask_spread = max(self.config.min_spread, min(self.config.max_spread, adj_ask_spread))

        buy_price = ref_price * (1 - adj_bid_spread)
        sell_price = ref_price * (1 + adj_ask_spread)

        self.logger().info(f"Spreads - Bid: {adj_bid_spread:.4%}, Ask: {adj_ask_spread:.4%}")
        self.logger().info(f"Volatility: {volatility:.4%}, Depth - Bid: {bid_ratio:.2}, Ask: {ask_ratio:.2}")
        self.logger().info(f"Prices - Buy: {buy_price}, Sell: {sell_price}")

        return [
            OrderCandidate(self.config.trading_pair, True, OrderType.LIMIT, TradeType.BUY,
                           self.config.order_amount, buy_price),
            OrderCandidate(self.config.trading_pair, True, OrderType.LIMIT, TradeType.SELL,
                           self.config.order_amount, sell_price)
        ]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connectors[self.config.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, proposal: List[OrderCandidate]):
        for order in proposal:
            self.place_order(self.config.exchange, order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        self.logger().info(f"Placing {'BUY' if order.order_side == TradeType.BUY else 'SELL'} "
                           f"{order.amount} {order.trading_pair} at {order.price:.6f}")
        if order.order_side == TradeType.SELL:
            self.sell(connector_name, order.trading_pair, order.amount, order.order_type, order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 8)} {event.trading_pair} {self.config.exchange} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        if self.entry_price is None:
            self.entry_price = event.price
            self.logger().info(f"Entry price set: {self.entry_price}")

        current_price = self.connectors[self.config.exchange].get_price_by_type(
            self.config.trading_pair, self.price_source)

        pnl = (current_price - self.entry_price) / self.entry_price if event.trade_type == TradeType.BUY \
            else (self.entry_price - current_price) / self.entry_price

        if pnl <= -self.config.stop_loss_pct:
            self.logger().info(f"Stop-loss hit: {pnl:.2%}")
            self.cancel_all_orders()

        elif pnl >= self.config.take_profit_pct:
            self.logger().info(f"Take-profit hit: {pnl:.2%}")
            self.cancel_all_orders()
        else:
            self.logger().info("No trades filled yet — monitoring continues.")

    def format_status(self) -> str:
        connector = self.connectors[self.config.exchange]
        mid_price = connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice)
        bid = connector.get_price_by_type(self.config.trading_pair, PriceType.BestBid)
        ask = connector.get_price_by_type(self.config.trading_pair, PriceType.BestAsk)

        if len(self.price_samples) > 1:
            mean_price = sum(self.price_samples) / len(self.price_samples)
            var = sum((p - mean_price) ** 2 for p in self.price_samples) / len(self.price_samples)
            volatility = (var ** 0.5 / mean_price) * 100
        else:
            volatility = 0

        # Get SMA for status
        sma = "N/A"
        if len(self.price_history) >= self.config.sma_period:
            sma = f"{np.mean(self.price_history[-self.config.sma_period:]):.4f}"
        
        # Get Bollinger Bands for status
        bb_info = "N/A"
        if len(self.price_history) >= self.config.bollinger_window:
            bb_mean = np.mean(self.price_history)
            bb_std = np.std(self.price_history)
            upper = bb_mean + float(self.config.bollinger_stddev) * bb_std
            lower = bb_mean - float(self.config.bollinger_stddev) * bb_std
            bb_info = f"[{lower:.4f}, {upper:.4f}]"

        base, quote = self.config.trading_pair.split("-")
        base_balance = connector.get_available_balance(base)
        quote_balance = connector.get_available_balance(quote)

        return "\n".join([
            "Strategy: ImprovedPMM",
            f"Exchange: {self.config.exchange}",
            f"Trading Pair: {self.config.trading_pair}",
            f"Mid Price: {mid_price:.8g}",
            f"Best Bid: {bid:.8g} | Best Ask: {ask:.8g}",
            f"Volatility: {volatility:.2f}%",
            f"SMA ({self.config.sma_period}): {sma}",
            f"Bollinger Bands: {bb_info}",
            f"Trend: {['Downtrend', 'Neutral', 'Uptrend'][self.price_movement_trend + 1]}",
            f"Active Orders: {len(self.get_active_orders(self.config.exchange))}",
            f"{base} Balance: {float(base_balance):.8f}",
            f"{quote} Balance: {float(quote_balance):.4f}"
        ])