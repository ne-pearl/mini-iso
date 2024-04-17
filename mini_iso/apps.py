from __future__ import annotations
import logging
import sys
import typing
import panel as pn
from mini_iso.auction import Auction
from mini_iso.bidders import Bidder
from mini_iso.dashboard import LmpPricer, LmpDashboard
from mini_iso.datasets.mini_new_england import load_system
from mini_iso.typing import Input

ADDRESS: typing.Final[str] = "*"
PORT: typing.Final[int] = 5000

# panel configuration
pn.extension(
    "tabulator",
    "vega",
    css_files=[
        # Required to render button icons
        # https://panel.holoviz.org/reference/widgets/Tabulator.html#buttons
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css"
    ],
    design="material",
    notifications=True,
    sizing_mode="stretch_width",
)
# Enables loading indicator globally
# https://panel.holoviz.org/how_to/param/examples/loading.html
pn.param.ParamMethod.loading_indicator = True

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)


@pn.cache
def load_auction(constrained: bool) -> Auction:
    """This function's body is evaluated exactly once."""
    inputs: Input = load_system(constrained=constrained)
    pricer: LmpPricer = LmpPricer.from_inputs(inputs)
    return Auction(pricer)


auction: Auction = load_auction(constrained=True)

# WARNING: The "Bidder_=Bidder" appears to be necessary in production.
# Without it, the code crashes inside somewhere in a JavaScript framework,
# complaining of a missing definition for class Bidder.
def new_bidding_session(auction=auction, Bidder_=Bidder):
    return Bidder_(auction)


if __name__ != "__main__":
    pn.serve(
        admin=True,
        panels={
            "iso-auction": auction,
            # "Back-End": auction.pricer,
            "generator-bidding": new_bidding_session,
            "system-dashboard": LmpDashboard(pricer=auction.pricer),
        },
        port=PORT,
        title="Mini-ISO: Application Menu",
        websocket_origin=f"{ADDRESS}:{PORT}",
    )

else:
    dashboard = LmpDashboard(pricer=auction.pricer)
    pn.panel(dashboard).servable()
