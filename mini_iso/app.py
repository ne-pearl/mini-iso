from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import textwrap
import panel as pn
from mini_iso.auction import Auction
from mini_iso.dashboard import LmpPricer, LmpDashboard

# from mini_iso.miscellaneous import DATASETS_ROOT_PATH
from mini_iso.miscellaneous import ADDRESS, PORT, DATASETS_ROOT_PATH
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
    inputs: Input = Input.from_json(
        case_path if case_path.is_absolute() else DATASETS_ROOT_PATH / case_path
    )
    pricer: LmpPricer = LmpPricer.from_inputs(inputs)
    return Auction(pricer)


parser = argparse.ArgumentParser(
    description="Mini-ISO Dashboard",
    epilog=textwrap.dedent(
        """\
        Names of embedded (internal) datasets:
        1. three-zones-case1
        2. three-zones-case2
        3. three-zones-case3
        4. three-zones-case4
        5. mini-new-england-uniform
        6. mini-new-england-nonuniform

        Example:
        $ mini_iso three-zones-case1

        Use Ctrl+C to exit the program.
        """
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("path", type=Path, help="Absolute path to external dataset OR name of embedded (internal) dataset")
parser.add_argument("--address", default=ADDRESS, help="IP address of host")
parser.add_argument("--port", default=PORT, help="Communication port")
args = parser.parse_args()
auction: Auction = load_auction(args.path)
dashboard = LmpDashboard(pricer=auction.pricer)
# pn.panel(dashboard).servable()

pn.serve(
    admin=True,
    panels=dashboard,
    port=args.port,
    title="Mini-ISO: Application Menu",
    websocket_origin=f"{args.address}:{args.port}",
)
