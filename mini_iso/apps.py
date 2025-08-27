from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import panel as pn
from mini_iso.auction import Auction
from mini_iso.bidders import Bidder
from mini_iso.dashboard import LmpPricer, LmpDashboard
from mini_iso.miscellaneous import PORT, DATASETS_ROOT_PATH
from mini_iso.typing_ import Input

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
)
# Enables loading indicator globally
# https://panel.holoviz.org/how_to/param/examples/loading.html
pn.param.ParamMethod.loading_indicator = True

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)


@pn.cache
def load_auction(case_path: Path) -> Auction:
    """This function's body is evaluated exactly once."""
    inputs: Input = Input.from_json(DATASETS_ROOT_PATH / case_path)
    pricer: LmpPricer = LmpPricer.from_inputs(inputs)
    return Auction(pricer)


parser = argparse.ArgumentParser(description="Mini-ISO applications")
parser.add_argument("path", type=Path)
args = parser.parse_args()
auction: Auction = load_auction(args.path)


# WARNING: The "Bidder_=Bidder" appears to be necessary in production.
# Without it, the code crashes inside somewhere in a JavaScript framework,
# complaining of a missing definition for class Bidder.
def new_bidding_session(auction=auction, Bidder_=Bidder):
    return Bidder_(auction)

dashboard = LmpDashboard(pricer=auction.pricer)

panels={
    "auction": auction,
    "back-end": auction.pricer,
    "offers": new_bidding_session,
    "app": dashboard,
}

# if __name__ == "__main__":
#     pn.serve(
#         admin=True,
#         panels=panels,
#         port=PORT,
#         title="Mini-ISO: Application Menu",
#         websocket_origin=f"*:{PORT}",
#     )

# else:
#     pn.panel(panels).servable()

pn.serve(
    admin=True,
    panels=panels,
    port=PORT,
    title="Mini-ISO: Application Menu",
    websocket_origin=f"*:{PORT}",
)
