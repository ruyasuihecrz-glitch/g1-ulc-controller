"""Open the substation USDZ once Isaac Sim has finished bringing up the GUI."""

import asyncio
import os

import omni.kit.app
import omni.usd
from pxr import Sdf


USDZ_PATH = os.environ.get("SUBSTATION_USDZ_PATH", "/workspace/data/substation_final_gs_mesh_collision.usdz")


async def _open_stage_after_startup():
    app = omni.kit.app.get_app()
    for _ in range(20):
        await app.next_update_async()

    print(f"[open_substation_stage] Opening stage: {USDZ_PATH}")
    omni.usd.get_context().open_stage(USDZ_PATH)

    for _ in range(40):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath("/World/ViewCamera").IsValid():
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            viewport = get_active_viewport()
            if viewport:
                viewport.camera_path = Sdf.Path("/World/ViewCamera")

                try:
                    import numpy as np
                    from isaacsim.core.utils.viewports import set_camera_view

                    set_camera_view(
                        eye=np.array([-5.8, -15.5, 5.2]),
                        target=np.array([-5.8, -4.4, 3.2]),
                        camera_prim_path="/World/ViewCamera",
                        viewport_api=viewport,
                    )
                except Exception as exc:
                    print(f"[open_substation_stage] Could not set camera view: {exc}")

                print("[open_substation_stage] Active viewport camera: /World/ViewCamera")

                for _ in range(120):
                    await app.next_update_async()

                capture_path = "/tmp/lcc2_isaac_view.png"
                await capture_viewport_to_file(
                    viewport, file_path=capture_path, is_hdr=False
                ).wait_for_result()
                print(f"[open_substation_stage] Viewport screenshot: {capture_path}")
        except Exception as exc:
            print(f"[open_substation_stage] Could not switch viewport camera: {exc}")


asyncio.ensure_future(_open_stage_after_startup())
