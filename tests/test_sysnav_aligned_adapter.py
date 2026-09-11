from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from sysnav_aligned_input import free_points, geometry, load_clouds, parameters, project_labels, project_doors


def test_native_free_footprint_never_expands_into_occupied_or_unknown():
    nav = np.zeros((12, 16), dtype=bool)
    nav[4:8, 4:8] = True
    bounds = [-.4, -.2, .4, .4]
    points = free_points(nav, bounds, .05, .1)
    assert len(points) == 1
    assert np.allclose(points[0, :2], [-.2, 0])
    # Mirror exactly native updateFreespace's forward 2x2 expansion.
    native_origin = np.floor((points[0, :2]+1e-4)/.1).astype(int)
    for dx in (0, 1):
        for dy in (0, 1):
            xy = (native_origin + [dx, dy])*.1
            col, row = np.rint((xy-np.array(bounds[:2]))/.05).astype(int)
            assert nav[row:row+2, col:col+2].all()


def test_partial_free_block_is_not_published():
    nav=np.ones((4,4), dtype=bool);nav[0,0]=False
    assert len(free_points(nav,[0,0,.2,.2],.05,0))==0
    assert len(free_points(np.ones((3,4),bool),[0,0,.2,.15],.05,0))==0


def test_asymmetric_world_projection_and_negative_coordinates():
    labels=np.arange(80,dtype=np.int32).reshape(8,10)
    meta={'source_shape_yx':[4,8],'source_resolution_m':.05,'source_bounds_xyxy_m':[-.3,-.2,.1,0]}
    projected=project_labels(labels,meta,.1)
    for row in range(4):
        for col in range(8):
            ix=int(np.floor((-.3+(col+.5)*.05)/.1+4))
            iy=int(np.floor((-.2+(row+.5)*.05)/.1+5))
            assert projected[row,col]==labels[ix,iy]
    meta['native_dimensions_xy']=[8,10]
    doors=project_doors(np.array([[-.2,-.1,0.]],dtype=np.float32),meta,.1)
    assert doors.sum()==4
    assert doors[2:4,2:4].all()


def test_observed_only_scan_no_fabricated_floor_or_room_labels(tmp_path):
    state=np.zeros((3,8,8),dtype=np.uint8)
    state[0,1:3,1:4]=2
    state[1,5:7,5:7]=2
    state[2,0,0]=2
    arrays={'voxel_occupancy_state_zyx':state,'voxel_occupancy_z_centers_m':np.array([0,.5,3.]),
            'voxel_nav_free_xy':np.ones((8,8),bool),'map_origin_xy_m':np.array([-.2,-.2]),
            'map_resolution_m':np.array(.05),'demo_pose_world':np.array([0,0,.05,0])}
    params={'ceilingHeight_':2.7,'room_resolution':.1,'room_x':3000,'room_y':3000,'room_z':80,
            'rolling_occupancy_grid.resolution_x':.2,'kViewPointCollisionMarginZMinus':.4,'kViewPointCollisionMarginZPlus':.75}
    p=tmp_path/'source.npz';np.savez(p,**arrays)
    clouds,meta,_=load_clouds(p,params)
    assert len(clouds['scan'])==10 and meta['synthetic_floor_point_count']==0
    assert set(clouds['scan'][:,2])=={0.,.5}
    assert np.all(clouds['occupied'][:,3]==0)
    arrays['final_room_label_map']=np.full((8,8),9999)
    arrays['door_seed_keep_mask']=np.ones((8,8),bool)
    arrays['roomseg_eval_explored_reference_mask']=np.zeros((8,8),bool)
    np.savez(p,**arrays)
    changed,_,_=load_clouds(p,params)
    for name in clouds:np.testing.assert_array_equal(clouds[name],changed[name])


@pytest.mark.skipif(not Path('/home/joey/SysNav').exists(),reason='Pinned upstream not installed')
def test_entire_official_sim_profile_preserved():
    params,profile=parameters(Path('/home/joey/SysNav'))
    for name,value in profile['original'].items():
        assert params[name]==(False if name=='isDebug' else value)
    assert params['region_growing_radius']==15
    assert (params['kViewPointCollisionMarginZMinus'],params['kViewPointCollisionMarginZPlus'])==(.4,.75)


def test_bridge_orders_native_callbacks_without_reinserting_points():
    text=(ROOT/'scripts/sysnav_snapshot_bridge.cpp').read_text()
    positions=[text.index('node->'+method) for method in
               ('laserCloudCallback','freespaceCloudCallback','occupiedCloudCallback','timerCallback')]
    assert positions==sorted(positions)
    assert text.count('node->laserCloudCallback')==1
    assert 'rclcpp::spin(' not in text


@pytest.mark.skipif(not (ROOT/'results/sysnav_aligned_20260911/inputs/frozen_tasks.json').exists(),reason='Staged test data unavailable')
def test_frozen_462_points_match_main_reference_and_exclude_training():
    out=ROOT/'results/sysnav_aligned_20260911'
    tasks=json.loads((out/'inputs/frozen_tasks.json').read_text())
    exclusions=set(json.loads((out/'inputs/excluded_scenes.json').read_text()))
    assert len(tasks)==462
    assert len({r['episode_uid'] for r in tasks})==68
    assert not {r['scene_id'] for r in tasks}&exclusions
    assert all(len(t['source_sha256'])==64 for t in tasks)
    # The unrelated native-DUDE failure must NOT silently remove this SysNav point.
    assert any(t['scene_id']=='kujiale_grscene_MVUHLWYKTKJ5EAABAAAAACI8_usd' and t['coverage_event_id']=='final' for t in tasks)


def test_independent_metrics_perfect_merged_and_empty_predictions():
    from verify_aligned_sysnav_results import independent_metrics
    gt=np.array([[1,1,2,2],[1,1,2,2]])
    explored=np.ones_like(gt,bool)
    perfect=independent_metrics(gt,explored,gt,1)
    assert perfect['f1']==perfect['miou_room']==1
    merged=independent_metrics(gt,explored,np.ones_like(gt),1)
    assert merged['precision']==.5 and merged['recall']==1
    assert merged['f1']==pytest.approx(2/3) and merged['miou_room']==.25
    empty=independent_metrics(gt,explored,np.zeros_like(gt),1)
    assert empty['f1']==empty['miou_room']==0


def test_independent_metric_domain_removes_small_gt_without_dropping_other_rooms():
    from verify_aligned_sysnav_results import independent_metrics
    gt=np.array([[1,1,1,2],[1,1,1,0]])
    result=independent_metrics(gt,np.ones_like(gt,bool),gt,2)
    assert result['n_gt']==result['n_pred']==1
    assert result['metric_domain_pixels']==6
    assert result['f1']==result['miou_room']==1
