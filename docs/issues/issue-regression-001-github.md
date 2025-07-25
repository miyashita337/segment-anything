# [Bug] Full-body character extraction regression in v0.4.0

## 🎯 Issue Summary

**Severity**: High - Critical functionality regression  
**Component**: Character extraction pipeline (SAM + YOLO)  
**Affected Version**: v0.4.0  
**Last Working Version**: v0.3.5

## 📋 Description

Character extraction system has regressed from extracting full-body anime characters to only extracting faces/heads after v0.4.0 release. This severely impacts the primary goal: "automatically extract full-body manga characters as completely as possible."

## 🔍 Current Behavior

- **v0.3.5**: Successfully extracted full-body characters with ~90% success rate
- **v0.4.0**: All output images contain only face/head regions, body parts are lost
- **Impact**: Complete failure to meet full-body extraction requirements

## ✅ Expected Behavior

- Extract complete character bodies including torso, limbs, and clothing
- Maintain v0.3.5-level success rate (90%+ full-body extractions)
- Achieve B-grade evaluation for 50%+ of outputs

## 🔧 Root Cause Analysis

### Primary Change in v0.4.0
- **Added**: P1-006 Solid Fill Detection feature
- **Files Modified**: 
  - `features/evaluation/utils/enhanced_solid_fill_processor.py`
  - `features/extraction/commands/extract_character.py`
  - `features/evaluation/utils/difficult_pose.py`

### Identified Issues

1. **Fixed Edge Margin Problem** (Critical)
   - Hard-coded 10% edge threshold classifies regions near image borders as 'background'
   - Manga panels often crop characters at frame borders
   - Body parts within 10% margin are incorrectly removed

2. **Over-specialized Classification Logic**
   - Only black regions with aspect ratio >1.5 classified as 'character' (hair-focused)
   - Clothing and body parts (especially with screentones) misclassified as 'background'

3. **Processing Order Issue**
   - Solid fill detection runs before SAM processing
   - Regions removed by solid fill cannot be recovered by SAM refinement

## 💡 Proposed Solution

### Approach: Fix classification logic while preserving P1-006 benefits

**Phase 1: Core Fixes**
- [ ] Replace fixed 10% edge threshold with adaptive margin: `max(min(24px, W×0.03), 4px)`
- [ ] Integrate YOLO/SAM detection info into region classification
- [ ] Implement weighted classification system instead of hard rules

**Phase 2: Fallback Mechanism**
- [ ] Add quality verification: if extracted height < 75% of YOLO height, retry without solid fill
- [ ] Select result with higher IoU overlap with YOLO detection

**Phase 3: Parameter Optimization**
- [ ] Adjust color uniformity thresholds: σ_L (3→6), σ_ab (4→8)
- [ ] Modify ranking algorithm: area×character_probability over boundary_quality alone

## 🧪 Acceptance Criteria

### Must-Have (All Required)
- [ ] Full-body extraction success rate ≥ 90%
- [ ] B-grade evaluation rate ≥ 50%
- [ ] Background false-positive rate ≤ 5%
- [ ] Processing time increase ≤ 50%

### Nice-to-Have (Optimization)
- [ ] A-grade evaluation rate ≥ 30%
- [ ] Processing time improvement
- [ ] Memory usage optimization

## 🔬 Test Plan

### Regression Test Dataset
- **Source**: kana08 dataset (26 images)
- **Baseline**: C:\AItools\lora\train\yado\clipped_boundingbox\kana08_0_4_0
- **Comparison**: v0.3.5 vs v0.4.0 vs v0.4.1 (fixed version)

### Test Execution
```bash
# Version comparison test
python tools/test_batch_extraction.py --dataset kana08 --compare-versions v0.3.5,v0.4.0,v0.4.1

# Quality metrics evaluation
python tools/evaluate_extraction_quality.py --results v0.4.1_results --metrics full_body,b_grade_rate
```

## 📊 Technical Details

### Files to Modify
1. `features/evaluation/utils/enhanced_solid_fill_processor.py`
   - `_classify_region_type()`: Add YOLO/SAM context
   - `_get_adaptive_edge_threshold()`: Replace fixed threshold
   
2. `features/extraction/commands/extract_character.py`
   - Add fallback mechanism for small extraction results
   - Pass YOLO detection info to solid fill processor

3. `features/evaluation/utils/difficult_pose.py`
   - Update `preprocess_for_difficult_pose()` to accept detection context

### Parameter Changes
```yaml
Edge Processing:
  edge_threshold: 0.10 → adaptive(0.03, cap=24px)

Color Uniformity:
  sigma_L_threshold: 3 → 6
  sigma_ab_threshold: 4 → 8

Region Classification:
  min_region_area: 0.005 → 0.002
  yolo_sam_prior_weight: 0.0 → 1.2 (new)
```

## 🏷️ Labels

`bug` `regression` `high-priority` `character-extraction` `solid-fill-detection`

## 🔗 Related Links

- **GitHub PR**: #8 (introduced the regression)
- **Technical Spec**: [issue-regression-001-spec.md](./issue-regression-001-spec.md)
- **AI Discussion**: [issue-regression-001-discussion.md](./issue-regression-001-discussion.md)

## 📅 Estimated Effort

- **Development**: 2.5 person-days
- **Testing**: 0.5 person-days
- **Total**: 3.0 person-days

## 👥 Assignment

- **Primary**: Claude Code + GPT-4O collaboration
- **Review**: Human evaluation (final approval)
- **Priority**: High (immediate fix required)