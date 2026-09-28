"""Editable visual task guidance, not control enforcement or ground-truth telemetry.

The release milestone describes an intermediate arrangement. Final success stays a
CURRENT-image judgment and does not require reconstructing the release history.
"""
import json
from .monitor import HERE
from .workspace import Task, new_id


def mujoco_plug_task(name='MuJoCo plug insertion', store=None):
    config = json.loads((HERE / 'config.json').read_text())
    config.update(base_url='https://api.openai.com/v1', model='gpt-6-luna', reasoning_effort='low', max_tokens=1024, max_in_flight=24)
    context = ('Monitor a Franka Panda moving an orange-handled plug with a silver cylindrical pin '
               'from a gray stand into the black bore of a tall blue socket. The pin and bore are round; '
               'yaw alone is not an insertion failure. The silver pin is 50 mm long and 20 mm wide. '
               'The intended controller sequence inserts at least 25 mm into the actual bore while '
               'the plug is still held, then intentionally releases it; the remainder may settle '
               'under gravity. This explains intermediate arrangements, not an extra final-success '
               'history requirement. Judge only CURRENT camera images. '
               'Camera views can be moved, so use physical relationships rather than fixed image coordinates. '
               'The COMPLETE reference has approximately 117 mm of TCP-to-handle-top clearance. That is a labeled reference example, NOT the minimum required gap; CURRENT needs only more than 50 mm (about 1.6 times the 32 mm handle height). A smaller gap than the reference can therefore still be successful. Never copy the reference measurement into CURRENT. The blue block is the DESTINATION SOCKET, never the source stand. The gray low base with a gray cylindrical peg is the SOURCE STAND. Its gray peg remains after the orange plug is lifted and is not the plug. Insertion fills the black bore, so the pin and opening may become hidden beneath the orange handle. If visibility is insufficient, abstain. No simulator success flags are provided.')
    specs = [
        ('success', 'Success', None, True,
         'Return POSITIVE only when the silver pin is fully seated in the black bore, the orange handle '
         'is supported by the socket, the gripper is open and no longer touches the plug, and its TCP '
         'is visibly more than 50 mm above the top of the handle. The handle is 32 mm tall for a visible '
         'scale cue. The TCP is the midpoint between the black fingertip pads, not the white palm. The COMPLETE reference is an accepted example of seating, release, and sufficient retreat. Compare its seating and release relations with CURRENT, then assess the CURRENT gap against 50 mm, not the larger reference gap. Do not reject a completed scene solely because its clearance is less than the reference. The gripper may remain directly above the socket and the arm may remain visible after retreat; that alone is not a negative. Require a suitable view and return UNCERTAIN if clearance cannot be established. '
         'Seated but held, hovering, partial insertion, or insufficient retreat are NEGATIVE.'),
        ('drop', 'Dropped plug', 'DROP_FAILURE', True,
         'Return POSITIVE only when the plug itself is visibly loose on the tabletop, separated from '
         'the gripper, off its source stand, and not seated in the socket. A plug still on the gray '
         'stand or released into the socket is NEGATIVE. Occlusion and an empty gripper alone do not '
         'prove a drop. A clearly supported plug on the gray stand or seated in the blue socket is NEGATIVE even if gripper contact is unclear. Use UNCERTAIN when the plug or its support cannot be established, or during an unresolved fall.'),
        ('slip', 'Slip or tilt (review before enabling)', 'SLIP_FAILURE', False,
         'Return POSITIVE only for visible loss of a secure handle-body grasp while still held: '
         'the handle has slipped between fingertip pads or the silver pin tilts relative to the '
         'gripper carry axis. Camera rotation and round-pin yaw alone are NEGATIVE. '
         'Do not infer a failure solely from approach misalignment or socket occlusion.')]
    specialists = []
    for key, label, failure, enabled, criterion in specs:
        prompt = context + '\n\n' + criterion + '\nReturn exactly JSON with verdict (POSITIVE, NEGATIVE, UNCERTAIN) and reason (one short sentence of visible evidence).'
        specialists.append({'id': key, 'name': label, 'enabled': enabled, 'threshold': 5,
                             'definition': {'field': 'verdict', 'values': {v: v for v in ('POSITIVE','NEGATIVE','UNCERTAIN')}, 'failure_type': failure},
                             'blocks': [{'id': key+'_context', 'type': 'text', 'text': prompt},
                                        {'id': key+'_view1', 'type': 'camera', 'source': 'camera1', 'alias': 'CURRENT overview', 'crop': None},
                                        {'id': key+'_view2', 'type': 'camera', 'source': 'camera2', 'alias': 'CURRENT inspection', 'crop': None}]})
    nodes = [
        ('ON_STAND', 'On stand', 'The plug remains supported by the gray source stand. The gripper is visibly open, separated from the handle, or approaching it. Open fingers on opposite sides of the handle still count as ON_STAND.'),
        ('GRASPED', 'Grasped', 'The plug remains at the source stand and opposing fingertip pads are visibly closed against both sides of the orange handle, with no visible contact gap. Merely surrounding or aligning with the handle is insufficient. If closure or contact is obscured, select UNKNOWN. This node describes a visible clamp, not proven grasp strength.'),
        ('LIFTED_CLEAR', 'Lifted clear', 'Plug is securely carried with its pin clear of the stand and blue socket top, but it is not yet aligned above the socket bore.'),
        ('ALIGNED_ABOVE_SOCKET', 'Aligned above socket', 'Held pin is visibly centered over the bore in both views with its tip above the bore rim; it has not entered.'),
        ('INSERTING', 'Entering the bore, still held', 'Closed fingertip pads still hold the handle and the pin has visibly entered the actual bore, but more than 25 mm of its 50 mm silver shaft remains above the rim. Projected overlap, a hidden tip, or a pin parallel behind the housing is not entry. Use UNKNOWN if engagement or exposed length cannot be judged.'),
        # Preserve the saved node ID; its label/criteria now express the requested release milestone.
        ('SEATED_HELD', 'Half inserted, still held', 'Opposing fingertip pads still hold the handle and at least 25 mm of the 50 mm pin is visibly inside the actual bore, with no more than 25 mm of silver shaft above the rim. Deeper or fully seated engagement while held also fits. Verify actual bore entry from the CURRENT external views, not a parallel shaft behind the housing or projected overlap. Use UNKNOWN if contact, engagement, or depth is unclear.'),
        ('RELEASED', 'Released', 'Plug remains seated; fingers are open and separated, but the TCP is not visibly more than 50 mm above handle top.'),
        ('RETREATED', 'Retreated', 'Plug remains seated and undisturbed; fingers are open and TCP is visibly more than 50 mm above the handle top.')]
    sources = []
    for number, azimuth in ((1, 45.), (2, 135.)):
        sources.append({'id': f'camera{number}', 'name': 'Overview' if number == 1 else 'Inspection',
                        'enabled': True, 'width': 640, 'height': 480, 'fps': 15,
                        'camera': {'type': 'free', 'lookat_m': [.4, 0., .16], 'distance_m': .85,
                                   'azimuth_deg': azimuth, 'elevation_deg': -25.}})
    task = Task.model_validate({'id': new_id('task'), 'name': name, 'config': config,
                                'input_mode': 'mujoco', 'sources': sources, 'specialists': specialists,
                                'simulation': {'base_url': 'http://127.0.0.1:8766', 'auto_stop': 'off', 'stale_after_s': 5.},
                                'progress': {'enabled': True, 'name': 'Plug insertion progress', 'max_tokens': 1024,
                                             'nodes': [{'id': key, 'label': label, 'criteria': criterion} for key,label,criterion in nodes],
                                             'blocks': [{'id': 'progress_context', 'type': 'text', 'text': context + '\nSelect UNKNOWN for ambiguous, occluded or dropped states. When the plug is on the stand, require visible opposing pad contact for GRASPED; open fingers surrounding the handle remain ON_STAND. Do not infer grasping from the expected task sequence or a grasp-ready pose. INSERTING is held entry less than halfway; SEATED_HELD retains its saved ID but means at least half inserted while still held, including deeper held insertion. Estimate exposure using the known 50 mm pin and 32 mm handle; hidden silver alone is not proof of entry. Released settling can be UNKNOWN until seating is visible. Do not use an inferred release history as an outcome vote. Nodes describe arrangements, not percentages.'},
                                                        {'id': 'progress_view1', 'type': 'camera', 'source': 'camera1', 'alias': 'CURRENT overview'},
                                                        {'id': 'progress_view2', 'type': 'camera', 'source': 'camera2', 'alias': 'CURRENT inspection'}]}}).model_dump()

    if store is not None:
        add_mujoco_references(store, task)
    return task


def add_mujoco_references(store, task):
    """Copy reviewed simulator examples into the normal task asset store."""
    examples = [('mujoco_initial.png', 'REFERENCE INITIAL: orange plug on GRAY source stand; NOT SUCCESS. The BLUE block is the empty destination socket.'),
                ('mujoco_completed.png', 'REFERENCE COMPLETE: orange plug seated in BLUE destination socket, gripper released and retreated with about 117 mm clearance. This is SUCCESS / RETREATED; the task minimum is only more than 50 mm, not 117 mm. The GRAY source stand is empty; its gray peg is not the plug.')]
    blocks = []
    for filename, label in examples:
        (store.assets / filename).write_bytes((HERE / 'references' / filename).read_bytes())
        blocks.append({'id': 'reference_' + filename.split('.')[0], 'type': 'image', 'alias': label,
                       'enabled': True, 'asset': filename, 'crop': None})
    for definition in [*task['specialists'], *([task['progress']] if task.get('progress') else [])]:
        existing = [b for b in definition['blocks'] if b.get('asset') not in {name for name, _ in examples}]
        definition['blocks'] = [*blocks, *existing]
    return task
