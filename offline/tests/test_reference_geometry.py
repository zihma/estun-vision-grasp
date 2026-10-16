import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_candidates import review, geometry_assumptions
class ReferenceGeometryTests(unittest.TestCase):
 def test_fallback_raises_tool_and_physical_tip_ten_mm(self):
  c={'rotation_matrix':[[0,1,0],[0,0,-1],[-1,0,0]],'position_xyz_m':[.4,0,.1],'insertion_depth_m':.02,'model_score':.9,'opening_width_m':.09}
  a=review(c,np.eye(4));b=review(c,np.eye(4),.160)
  for field in ['estimated_tool0_base_m','estimated_tip_base_m','estimated_pregrasp_tool0_base_m']:
   np.testing.assert_allclose(np.array(b[field])-a[field],[0,0,.01],atol=1e-9)
  np.testing.assert_allclose(a['grasp_reference_base_m'],b['grasp_reference_base_m'])
  self.assertEqual(a['equivalent_tool0_quaternions_xyzw'],b['equivalent_tool0_quaternions_xyzw'])
  self.assertEqual(geometry_assumptions()['flange_to_grasp_reference_m'],.150)
  self.assertEqual(geometry_assumptions(.160)['physical_flange_to_tip_m'],.180)
 def test_bad_reference_rejected(self):
  for value in [.149,.181,float('nan')]:
   with self.assertRaises(ValueError):review({},np.eye(4),value)
if __name__=='__main__':unittest.main()
