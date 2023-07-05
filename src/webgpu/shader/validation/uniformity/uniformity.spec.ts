export const description = `Validation tests for uniformity analysis`;

import { makeTestGroup } from '../../../../common/framework/test_group.js';
import { keysOf } from '../../../../common/util/data_tables.js';
import { unreachable } from '../../../../common/util/util.js';
import { ShaderValidationTest } from '../shader_validation_test.js';

export const g = makeTestGroup(ShaderValidationTest);

const kCollectiveOps = [
  { op: 'textureSample', stage: 'fragment' },
  { op: 'textureSampleBias', stage: 'fragment' },
  { op: 'textureSampleCompare', stage: 'fragment' },
  { op: 'dpdx', stage: 'fragment' },
  { op: 'dpdxCoarse', stage: 'fragment' },
  { op: 'dpdxFine', stage: 'fragment' },
  { op: 'dpdy', stage: 'fragment' },
  { op: 'dpdyCoarse', stage: 'fragment' },
  { op: 'dpdyFine', stage: 'fragment' },
  { op: 'fwidth', stage: 'fragment' },
  { op: 'fwidthCoarse', stage: 'fragment' },
  { op: 'fwidthFine', stage: 'fragment' },
  { op: 'storageBarrier', stage: 'compute' },
  { op: 'workgroupBarrier', stage: 'compute' },
  { op: 'workgroupUniformLoad', stage: 'compute' },
];

const kConditions = [
  { cond: 'uniform_storage_ro', expectation: true },
  { cond: 'nonuniform_storage_ro', expectation: false },
  { cond: 'nonuniform_storage_rw', expectation: false },
  { cond: 'nonuniform_builtin', expectation: false },
  { cond: 'uniform_literal', expectation: true },
  { cond: 'uniform_const', expectation: true },
  { cond: 'uniform_override', expectation: true },
  { cond: 'uniform_let', expectation: true },
  { cond: 'nonuniform_let', expectation: false },
  { cond: 'uniform_or', expectation: true },
  { cond: 'nonuniform_or1', expectation: false },
  { cond: 'nonuniform_or2', expectation: false },
  { cond: 'uniform_and', expectation: true },
  { cond: 'nonuniform_and1', expectation: false },
  { cond: 'nonuniform_and2', expectation: false },
  { cond: 'uniform_func_var', expectation: true },
  { cond: 'nonuniform_func_var', expectation: false },
];

function generateCondition(condition: string): string {
  switch (condition) {
    case 'uniform_storage_ro': {
      return `ro_buffer[0] == 0`;
    }
    case 'nonuniform_storage_ro': {
      return `ro_buffer[priv_var[0]] == 0`;
    }
    case 'nonuniform_storage_rw': {
      return `rw_buffer[0] == 0`;
    }
    case 'nonuniform_builtin': {
      return `p.x == 0`;
    }
    case 'uniform_literal': {
      return `false`;
    }
    case 'uniform_const': {
      return `c`;
    }
    case 'uniform_override': {
      return `o == 0`;
    }
    case 'uniform_let': {
      return `u_let == 0`;
    }
    case 'nonuniform_let': {
      return `n_let == 0`;
    }
    case 'uniform_or': {
      return `u_let == 0 || uniform_buffer.y > 1`;
    }
    case 'nonuniform_or1': {
      return `u_let == 0 || n_let == 0`;
    }
    case 'nonuniform_or2': {
      return `n_let == 0 || u_let == 0`;
    }
    case 'uniform_and': {
      return `u_let == 0 && uniform_buffer.y > 1`;
    }
    case 'nonuniform_and1': {
      return `u_let == 0 && n_let == 0`;
    }
    case 'nonuniform_and2': {
      return `n_let == 0 && u_let == 0`;
    }
    case 'uniform_func_var': {
      return `u_f == 0`;
    }
    case 'nonuniform_func_var': {
      return `n_f == 0`;
    }
    default: {
      unreachable(`Unhandled condition`);
    }
  }
}

function generateOp(op: string): string {
  switch (op) {
    case 'textureSample': {
      return `let x = ${op}(tex, s, vec2(0,0));\n`;
    }
    case 'textureSampleBias': {
      return `let x = ${op}(tex, s, vec2(0,0), 0);\n`;
    }
    case 'textureSampleCompare': {
      return `let x = ${op}(tex_depth, s_comp, vec2(0,0), 0);\n`;
    }
    case 'storageBarrier':
    case 'workgroupBarrier': {
      return `${op}();\n`;
    }
    case 'workgroupUniformLoad': {
      return `let x = ${op}(&wg);`;
    }
    case 'dpdx':
    case 'dpdxCoarse':
    case 'dpdxFine':
    case 'dpdy':
    case 'dpdyCoarse':
    case 'dpdyFine':
    case 'fwidth':
    case 'fwidthCoarse':
    case 'fwidthFine': {
      return `let x = ${op}(0);\n`;
    }
    default: {
      unreachable(`Unhandled op`);
    }
  }
}

function generateConditionalStatement(statement: string, condition: string, op: string): string {
  const code = ``;
  switch (statement) {
    case 'if': {
      return `if ${generateCondition(condition)} {
        ${generateOp(op)};
      }
      `;
    }
    case 'for': {
      return `for (; ${generateCondition(condition)};) {
        ${generateOp(op)};
      }
      `;
    }
    case 'while': {
      return `while ${generateCondition(condition)} {
        ${generateOp(op)};
      }
      `;
    }
    case 'switch': {
      return `switch u32(${generateCondition(condition)}) {
        case 0: {
          ${generateOp(op)};
        }
        default: { }
      }
      `;
    }
    default: {
      unreachable(`Unhandled statement`);
    }
  }

  return code;
}

g.test('basics')
  .desc(`Test collective operations in simple uniform or non-uniform control flow.`)
  .params(u =>
    u
      .combineWithParams(kCollectiveOps)
      .combineWithParams(kConditions)
      .combine('statement', ['if', 'for', 'while', 'switch'] as const)
      .beginSubcases()
  )
  .fn(t => {
    let code = `
 @group(0) @binding(0) var s : sampler;
 @group(0) @binding(1) var s_comp : sampler_comparison;
 @group(0) @binding(2) var tex : texture_2d<f32>;
 @group(0) @binding(3) var tex_depth : texture_depth_2d;

 @group(1) @binding(0) var<storage, read> ro_buffer : array<f32, 4>;
 @group(1) @binding(1) var<storage, read_write> rw_buffer : array<f32, 4>;
 @group(1) @binding(2) var<uniform> uniform_buffer : vec4<f32>;

 var<private> priv_var : array<f32, 4> = array(0,0,0,0);

 const c = false;
 override o : f32;
`;

    if (t.params.stage === 'compute') {
      code += `var<workgroup> wg : f32;\n`;
      code += ` @workgroup_size(16, 1, 1)`;
    }
    code += `@${t.params.stage}`;
    code += `\nfn main(`;
    if (t.params.stage === 'compute') {
      code += `@builtin(global_invocation_id) p : vec3<u32>`;
    } else {
      code += `@builtin(position) p : vec4<f32>`;
    }
    code += `) {
      let u_let = uniform_buffer.x;
      let n_let = rw_buffer[0];
      var u_f = uniform_buffer.z;
      var n_f = rw_buffer[1];
    `;

    // Simple control statement containing the op.
    code += generateConditionalStatement(t.params.statement, t.params.cond, t.params.op);

    code += `\n}\n`;

    t.expectCompileResult(t.params.expectation, code);
  });

const kFragmentBuiltinValues = [
  {
    builtin: `position`,
    type: `vec4<f32>`,
  },
  {
    builtin: `front_facing`,
    type: `bool`,
  },
  {
    builtin: `sample_index`,
    type: `u32`,
  },
  {
    builtin: `sample_mask`,
    type: `u32`,
  },
];

g.test('fragment_builtin_values')
  .desc(`Test uniformity of fragment built-in values`)
  .params(u => u.combineWithParams(kFragmentBuiltinValues).beginSubcases())
  .fn(t => {
    let cond = ``;
    switch (t.params.type) {
      case `u32`:
      case `i32`:
      case `f32`: {
        cond = `p > 0`;
        break;
      }
      case `vec4<u32>`:
      case `vec4<i32>`:
      case `vec4<f32>`: {
        cond = `p.x > 0`;
        break;
      }
      case `bool`: {
        cond = `p`;
        break;
      }
      default: {
        unreachable(`Unhandled type`);
      }
    }
    const code = `
@group(0) @binding(0) var s : sampler;
@group(0) @binding(1) var tex : texture_2d<f32>;

@fragment
fn main(@builtin(${t.params.builtin}) p : ${t.params.type}) {
  if ${cond} {
    let texel = textureSample(tex, s, vec2<f32>(0,0));
  }
}
`;

    t.expectCompileResult(true, `diagnostic(off, derivative_uniformity);\n` + code);
    t.expectCompileResult(false, code);
  });

const kComputeBuiltinValues = [
  {
    builtin: `local_invocation_id`,
    type: `vec3<f32>`,
    uniform: false,
  },
  {
    builtin: `local_invocation_index`,
    type: `u32`,
    uniform: false,
  },
  {
    builtin: `global_invocation_id`,
    type: `vec3<u32>`,
    uniform: false,
  },
  {
    builtin: `workgroup_id`,
    type: `vec3<u32>`,
    uniform: true,
  },
  {
    builtin: `num_workgroups`,
    type: `vec3<u32>`,
    uniform: true,
  },
];

g.test('compute_builtin_values')
  .desc(`Test uniformity of compute built-in values`)
  .params(u => u.combineWithParams(kComputeBuiltinValues).beginSubcases())
  .fn(t => {
    let cond = ``;
    switch (t.params.type) {
      case `u32`:
      case `i32`:
      case `f32`: {
        cond = `p > 0`;
        break;
      }
      case `vec3<u32>`:
      case `vec3<i32>`:
      case `vec3<f32>`: {
        cond = `p.x > 0`;
        break;
      }
      case `bool`: {
        cond = `p`;
        break;
      }
      default: {
        unreachable(`Unhandled type`);
      }
    }
    const code = `
@compute @workgroup_size(16,1,1)
fn main(@builtin(${t.params.builtin}) p : ${t.params.type}) {
  if ${cond} {
    workgroupBarrier();
  }
}
`;

    t.expectCompileResult(t.params.uniform, code);
  });

function generatePointerCheck(check: string): string {
  if (check === `address`) {
    return `let tmp = workgroupUniformLoad(ptr);`;
  } else {
    // check === `contents`
    return `if test_val > 0 {
      workgroupBarrier();
    }`;
  }
}

const kPointerCases = {
  address_uniform_literal: {
    code: `let ptr = &wg_array[0];`,
    check: `address`,
    uniform: true,
  },
  address_uniform_value: {
    code: `let ptr = &wg_array[uniform_value];`,
    check: `address`,
    uniform: true,
  },
  address_nonuniform_value: {
    code: `let ptr = &wg_array[nonuniform_value];`,
    check: `address`,
    uniform: false,
  },
  address_uniform_chain: {
    code: `let p1 = &wg_struct.x;
    let p2 = &(*p1)[uniform_value];
    let p3 = &(*p2).x;
    let ptr = &(*p3)[uniform_value];`,
    check: `address`,
    uniform: true,
  },
  address_nonuniform_chain1: {
    code: `let p1 = &wg_struct.x;
    let p2 = &(*p1)[nonuniform_value];
    let p3 = &(*p2).x;
    let ptr = &(*p3)[uniform_value];`,
    check: `address`,
    uniform: false,
  },
  address_nonuniform_chain2: {
    code: `let p1 = &wg_struct.x;
    let p2 = &(*p1)[uniform_value];
    let p3 = &(*p2).x;
    let ptr = &(*p3)[nonuniform_value];`,
    check: `address`,
    uniform: false,
  },
  wg_uniform_load_is_uniform: {
    code: `let test_val = workgroupUniformLoad(&wg_scalar);`,
    check: `contents`,
    uniform: true,
  },
  contents_scalar_uniform1: {
    code: `let ptr = &func_scalar;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: true,
  },
  contents_scalar_uniform2: {
    code: `func_scalar = nonuniform_value;
    let ptr = &func_scalar;
    func_scalar = 0;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: true,
  },
  contents_scalar_uniform3: {
    code: `let ptr = &func_scalar;
    func_scalar = nonuniform_value;
    func_scalar = uniform_value;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: true,
  },
  contents_scalar_nonuniform1: {
    code: `func_scalar = nonuniform_value;
    let ptr = &func_scalar;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_scalar_nonuniform2: {
    code: `let ptr = &func_scalar;
    *ptr = nonuniform_value;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_scalar_alias_uniform: {
    code: `let p = &func_scalar;
    let ptr = p;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: true,
  },
  contents_scalar_alias_nonuniform1: {
    code: `func_scalar = nonuniform_value;
    let p = &func_scalar;
    let ptr = p;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_scalar_alias_nonuniform2: {
    code: `let p = &func_scalar;
    *p = nonuniform_value;
    let ptr = p;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_scalar_alias_nonuniform3: {
    code: `let p = &func_scalar;
    let ptr = p;
    *p = nonuniform_value;
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_scalar_alias_nonuniform4: {
    code: `let p = &func_scalar;
    func_scalar = nonuniform_value;
    let test_val = *p;`,
    check: `contents`,
    uniform: false,
  },
  contents_scalar_alias_nonuniform5: {
    code: `let p = &func_scalar;
    *p = nonuniform_value;
    let test_val = func_scalar;`,
    check: `contents`,
    uniform: false,
  },
  contents_array_uniform_index: {
    code: `let ptr = &func_array[uniform_value];
    let test_val = *ptr;`,
    check: `contents`,
    uniform: true,
  },
  contents_array_nonuniform_index1: {
    code: `let ptr = &func_array[nonuniform_value];
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_array_nonuniform_index2: {
    code: `let ptr = &func_array[lid.x];
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_array_nonuniform_index3: {
    code: `let ptr = &func_array[gid.x];
    let test_val = *ptr;`,
    check: `contents`,
    uniform: false,
  },
  contents_struct_uniform: {
    code: `let p1 = &func_struct.x[uniform_value].x[uniform_value].x[uniform_value];
    let test_val = *p1;`,
    check: `contents`,
    uniform: true,
  },
  contents_struct_nonuniform1: {
    code: `let p1 = &func_struct.x[nonuniform_value].x[uniform_value].x[uniform_value];
    let test_val = *p1;`,
    check: `contents`,
    uniform: false,
  },
  contents_struct_nonuniform2: {
    code: `let p1 = &func_struct.x[uniform_value].x[gid.x].x[uniform_value];
    let test_val = *p1;`,
    check: `contents`,
    uniform: false,
  },
  contents_struct_nonuniform3: {
    code: `let p1 = &func_struct.x[uniform_value].x[uniform_value].x[lid.y];
    let test_val = *p1;`,
    check: `contents`,
    uniform: false,
  },
  contents_struct_chain_uniform: {
    code: `let p1 = &func_struct.x;
    let p2 = &(*p1)[uniform_value];
    let p3 = &(*p2).x;
    let p4 = &(*p3)[uniform_value];
    let p5 = &(*p4).x;
    let p6 = &(*p5)[uniform_value];
    let test_val = *p6;`,
    check: `contents`,
    uniform: true,
  },
  contents_struct_chain_nonuniform1: {
    code: `let p1 = &func_struct.x;
    let p2 = &(*p1)[nonuniform_value];
    let p3 = &(*p2).x;
    let p4 = &(*p3)[uniform_value];
    let p5 = &(*p4).x;
    let p6 = &(*p5)[uniform_value];
    let test_val = *p6;`,
    check: `contents`,
    uniform: false,
  },
  contents_struct_chain_nonuniform2: {
    code: `let p1 = &func_struct.x;
    let p2 = &(*p1)[uniform_value];
    let p3 = &(*p2).x;
    let p4 = &(*p3)[gid.x];
    let p5 = &(*p4).x;
    let p6 = &(*p5)[uniform_value];
    let test_val = *p6;`,
    check: `contents`,
    uniform: false,
  },
  contents_struct_chain_nonuniform3: {
    code: `let p1 = &func_struct.x;
    let p2 = &(*p1)[uniform_value];
    let p3 = &(*p2).x;
    let p4 = &(*p3)[uniform_value];
    let p5 = &(*p4).x;
    let p6 = &(*p5)[lid.y];
    let test_val = *p6;`,
    check: `contents`,
    uniform: false,
  },
};

g.test('pointers')
  .desc(`Test pointer uniformity (contents and addresses)`)
  .params(u => u.combine('case', keysOf(kPointerCases)).beginSubcases())
  .fn(t => {
    const testcase = kPointerCases[t.params.case];
    const code = `
var<workgroup> wg_scalar : u32;
var<workgroup> wg_array : array<u32, 16>;

struct Inner {
  x : array<u32, 4>
}
struct Middle {
  x : array<Inner, 4>
}
struct Outer {
  x : array<Middle, 4>
}
var<workgroup> wg_struct : Outer;

@group(0) @binding(0)
var<storage> uniform_value : u32;
@group(0) @binding(1)
var<storage, read_write> nonuniform_value : u32;

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(global_invocation_id) gid : vec3<u32>) {
  var func_scalar : u32;
  var func_array : array<u32, 16>;
  var func_struct : Outer;

  ${testcase.code}
`;

    const with_check =
      code +
      `
${generatePointerCheck(testcase.check)}
}`;
    if (!testcase.uniform) {
      const without_check = code + `}\n`;
      t.expectCompileResult(true, without_check);
    }
    t.expectCompileResult(testcase.uniform, with_check);
  });

function expectedUniformity(uniform: string, init: string): boolean {
  if (uniform === `always`) {
    return true;
  } else if (uniform === `init`) {
    return init === `no_init` || init === `uniform`;
  }

  // uniform == `never` (or unknown values)
  return false;
}

const kFuncVarCases = {
  no_assign: {
    typename: `u32`,
    typedecl: ``,
    assignment: ``,
    cond: `x > 0`,
    uniform: `init`,
  },
  simple_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `x = uniform_value[0];`,
    cond: `x > 0`,
    uniform: `always`,
  },
  simple_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `x = nonuniform_value[0];`,
    cond: `x > 0`,
    uniform: `never`,
  },
  compound_assign_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `x += uniform_value[0];`,
    cond: `x > 0`,
    uniform: `init`,
  },
  compound_assign_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `x -= nonuniform_value[0];`,
    cond: `x > 0`,
    uniform: `never`,
  },
  unreachable_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      break;
      x = uniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  unreachable_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      break;
      x = nonuniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  if_no_else_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = uniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  if_no_else_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = nonuniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  if_no_then_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
    } else {
      x = uniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  if_no_then_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
    } else {
      x = nonuniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  if_else_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = uniform_value[0];
    } else {
      x = uniform_value[1];
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  if_else_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = nonuniform_value[0];
    } else {
      x = nonuniform_value[1];
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  if_else_split: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = uniform_value[0];
    } else {
      x = nonuniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  if_unreachable_else_none: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
    } else {
      return;
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  if_unreachable_else_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = uniform_value[0];
    } else {
      return;
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  if_unreachable_else_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = nonuniform_value[0];
    } else {
      return;
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  if_unreachable_then_none: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      return;
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  if_unreachable_then_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      return;
    } else {
      x = uniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  if_unreachable_then_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      return;
    } else {
      x = nonuniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  if_nonescaping_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `if uniform_cond {
      x = nonuniform_value[0];
      return;
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  loop_body_depends_on_continuing_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if x > 0 {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
      continuing {
        x = uniform_value[0];
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `init`,
  },
  loop_body_depends_on_continuing_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if x > 0 {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
      continuing {
        x = nonuniform_value[0];
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `never`,
  },
  loop_body_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      x = uniform_value[0];
      continuing {
        break if uniform_cond;
      }
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  loop_body_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      x = nonuniform_value[0];
      continuing {
        break if uniform_cond;
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  loop_body_nonuniform_cond: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      // The analysis doesn't recognize the content of the value.
      x = uniform_value[0];
      continuing {
        break if nonuniform_cond;
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  loop_unreachable_continuing: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      break;
      continuing {
        break if uniform_cond;
      }
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  loop_continuing_from_body_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      x = uniform_value[0];
      continuing  {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `always`,
  },
  loop_continuing_from_body_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      x = nonuniform_value[0];
      continuing  {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `never`,
  },
  loop_continuing_from_body_split1: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if uniform_cond {
        x = uniform_value[0];
      }
      continuing {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `init`,
  },
  loop_continuing_from_body_split2: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if uniform_cond {
        x = nonuniform_value[0];
      }
      continuing {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `never`,
  },
  loop_continuing_from_body_split3: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if uniform_cond {
        x = uniform_value[0];
      } else {
        x = uniform_value[1];
      }
      continuing {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `always`,
  },
  loop_continuing_from_body_split4: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if nonuniform_cond {
        x = uniform_value[0];
      } else {
        x = uniform_value[1];
      }
      continuing {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `never`,
  },
  loop_continuing_from_body_split5: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if nonuniform_cond {
        x = uniform_value[0];
      } else {
        x = uniform_value[0];
      }
      continuing {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    // The analysis doesn't recognize that uniform_value[0] is assignment on all paths.
    uniform: `never`,
  },
  loop_in_loop_with_continue_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      loop {
        x = nonuniform_value[0];
        if nonuniform_cond {
          break;
        }
        continue;
      }
      x = uniform_value[0];
      continuing {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `always`,
  },
  loop_in_loop_with_continue_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      loop {
        x = uniform_value[0];
        if uniform_cond {
          break;
        }
        continue;
      }
      x = nonuniform_value[0];
      continuing {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
        break if uniform_cond;
      }
    }`,
    cond: `true`, // override the standard check
    uniform: `never`,
  },
  after_loop_with_uniform_break_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if uniform_cond {
        x = uniform_value[0];
        break;
      }
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  after_loop_with_uniform_break_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if uniform_cond {
        x = nonuniform_value[0];
        break;
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  after_loop_with_nonuniform_break: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if nonuniform_cond {
        x = uniform_value[0];
        break;
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  after_loop_with_uniform_breaks: {
    typename: `u32`,
    typedecl: ``,
    assignment: `loop {
      if uniform_cond {
        x = uniform_value[0];
        break;
      } else {
        break;
      }
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  switch_uniform_case: {
    typename: `u32`,
    typedecl: ``,
    assignment: `switch uniform_val {
      case 0 {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
      }
      default {
      }
    }`,
    cond: `true`, // override default check
    uniform: `init`,
  },
  switch_nonuniform_case: {
    typename: `u32`,
    typedecl: ``,
    assignment: `switch nonuniform_val {
      case 0 {
        if x > 0 {
          let tmp = textureSample(t, s, vec2f(0,0));
        }
      }
      default {
      }
    }`,
    cond: `true`, // override default check
    uniform: `never`,
  },
  after_switch_all_uniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `switch uniform_val {
      case 0 {
        x = uniform_value[0];
      }
      case 1,2 {
        x = uniform_value[1];
      }
      default {
        x = uniform_value[2];
      }
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  after_switch_some_assign: {
    typename: `u32`,
    typedecl: ``,
    assignment: `switch uniform_val {
      case 0 {
        x = uniform_value[0];
      }
      case 1,2 {
        x = uniform_value[1];
      }
      default {
      }
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  after_switch_nonuniform: {
    typename: `u32`,
    typedecl: ``,
    assignment: `switch uniform_val {
      case 0 {
        x = uniform_value[0];
      }
      case 1,2 {
        x = uniform_value[1];
      }
      default {
        x = nonuniform_value[0];
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  after_switch_with_break_nonuniform1: {
    typename: `u32`,
    typedecl: ``,
    assignment: `switch uniform_val {
      default {
        if uniform_cond {
          x = uniform_value[0];
          break;
        }
        x = nonuniform_value[0];
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  after_switch_with_break_nonuniform2: {
    typename: `u32`,
    typedecl: ``,
    assignment: `switch uniform_val {
      default {
        x = uniform_value[0];
        if uniform_cond {
          x = nonuniform_value[0];
          break;
        }
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  for_loop_uniform_body: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (var i = 0; i < 10; i += 1) {
      x = uniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  for_loop_nonuniform_body: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (var i = 0; i < 10; i += 1) {
      x = nonuniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  for_loop_uniform_body_no_condition: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (var i = 0; ; i += 1) {
      x = uniform_value[0];
      if uniform_cond {
        break;
      }
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  for_loop_nonuniform_body_no_condition: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (var i = 0; ; i += 1) {
      x = nonuniform_value[0];
      if uniform_cond {
        break;
      }
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  for_loop_uniform_increment: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (; uniform_cond; x += uniform_value[0]) {
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  for_loop_nonuniform_increment: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (; uniform_cond; x += nonuniform_value[0]) {
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  for_loop_uniform_init: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (x = uniform_value[0]; uniform_cond; ) {
    }`,
    cond: `x > 0`,
    uniform: `always`,
  },
  for_loop_nonuniform_init: {
    typename: `u32`,
    typedecl: ``,
    assignment: `for (x = nonuniform_value[0]; uniform_cond;) {
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  while_loop_uniform_body: {
    typename: `u32`,
    typedecl: ``,
    assignment: `while uniform_cond {
      x = uniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `init`,
  },
  while_loop_nonuniform_body: {
    typename: `u32`,
    typedecl: ``,
    assignment: `while uniform_cond {
      x = nonuniform_value[0];
    }`,
    cond: `x > 0`,
    uniform: `never`,
  },
  partial_assignment_uniform: {
    typename: `block`,
    typedecl: `struct block {
      x : u32,
      y : u32
    }`,
    assignment: `x.x = uniform_value[0].x;`,
    cond: `x.x > 0`,
    uniform: `init`,
  },
  partial_assignment_nonuniform: {
    typename: `block`,
    typedecl: `struct block {
      x : u32,
      y : u32
    }`,
    assignment: `x.x = nonuniform_value[0].x;`,
    cond: `x.x > 0`,
    uniform: `never`,
  },
  partial_assignment_all_members_uniform: {
    typename: `block`,
    typedecl: `struct block {
      x : u32,
      y : u32
    }`,
    assignment: `x.x = uniform_value[0].x;
    x.y = uniform_value[1].y;`,
    cond: `x.x > 0`,
    uniform: `init`,
  },
  partial_assignment_all_members_nonuniform: {
    typename: `block`,
    typedecl: `struct block {
      x : u32,
      y : u32
    }`,
    assignment: `x.x = nonuniform_value[0].x;
    x.y = uniform_value[0].x;`,
    cond: `x.x > 0`,
    uniform: `never`,
  },
  partial_assignment_single_element_struct_uniform: {
    typename: `block`,
    typedecl: `struct block {
      x : u32
    }`,
    assignment: `x.x = uniform_value[0].x;`,
    cond: `x.x > 0`,
    uniform: `init`,
  },
  partial_assignment_single_element_struct_nonuniform: {
    typename: `block`,
    typedecl: `struct block {
      x : u32
    }`,
    assignment: `x.x = nonuniform_value[0].x;`,
    cond: `x.x > 0`,
    uniform: `never`,
  },
  partial_assignment_single_element_array_uniform: {
    typename: `array<u32, 1>`,
    typedecl: ``,
    assignment: `x[0] = uniform_value[0][0];`,
    cond: `x[0] > 0`,
    uniform: `init`,
  },
  partial_assignment_single_element_array_nonuniform: {
    typename: `array<u32, 1>`,
    typedecl: ``,
    assignment: `x[0] = nonuniform_value[0][0];`,
    cond: `x[0] > 0`,
    uniform: `never`,
  },
  nested1: {
    typename: `block`,
    typedecl: `struct block {
      x : u32,
      y : u32
    }`,
    assignment: `for (; uniform_cond; ) {
      if uniform_cond {
        x = uniform_value[0];
        break;
        x.y = nonuniform_value[0].y;
      } else {
        if uniform_cond {
          continue;
        }
        x = uniform_value[1];
      }
    }`,
    cond: `x.x > 0`,
    uniform: `init`,
  },
  nested2: {
    typename: `block`,
    typedecl: `struct block {
      x : u32,
      y : u32
    }`,
    assignment: `for (; uniform_cond; ) {
      if uniform_cond {
        x = uniform_value[0];
        break;
        x.y = nonuniform_value[0].y;
      } else {
        if nonuniform_cond {
          continue;
        }
        x = uniform_value[1];
      }
    }`,
    cond: `x.x > 0`,
    uniform: `never`,
  },
};

const kVarInit = {
  no_init: ``,
  uniform: `= uniform_value[3];`,
  nonuniform: `= nonuniform_value[3];`,
};

g.test('function_variables')
  .desc(`Test uniformity of function variables`)
  .params(u => u.combine('case', keysOf(kFuncVarCases)).combine('init', keysOf(kVarInit)))
  .fn(t => {
    const func_case = kFuncVarCases[t.params.case];
    const code = `
${func_case.typedecl}

@group(0) @binding(0)
var<storage> uniform_value : array<${func_case.typename}, 4>;
@group(0) @binding(1)
var<storage, read_write> nonuniform_value : array<${func_case.typename}, 4>;

@group(1) @binding(0)
var t : texture_2d<f32>;
@group(1) @binding(1)
var s : sampler;

var<private> nonuniform_cond : bool = true;
const uniform_cond : bool = true;
var<private> nonuniform_val : u32 = 0;
const uniform_val : u32 = 0;

@fragment
fn main() {
  var x : ${func_case.typename} ${kVarInit[t.params.init]};

  ${func_case.assignment}

  if ${func_case.cond} {
    let tmp = textureSample(t, s, vec2f(0,0));
  }
}
`;

    const result = expectedUniformity(func_case.uniform, t.params.init);
    if (!result) {
      t.expectCompileResult(true, `diagnostic(off, derivative_uniformity);\n` + code);
    }
    t.expectCompileResult(result, code);
  });

const kShortCircuitExpressionCases = {
  or_uniform_uniform: {
    code: `
      let x = uniform_cond || uniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: true,
  },
  or_uniform_nonuniform: {
    code: `
      let x = uniform_cond || nonuniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: false,
  },
  or_nonuniform_uniform: {
    code: `
      let x = nonuniform_cond || uniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: false,
  },
  or_nonuniform_nonuniform: {
    code: `
      let x = nonuniform_cond || nonuniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: false,
  },
  or_uniform_first_nonuniform: {
    code: `
      let x = textureSample(t, s, vec2f(0,0)).x == 0 || nonuniform_cond;
    `,
    uniform: true,
  },
  or_uniform_second_nonuniform: {
    code: `
      let x = nonuniform_cond || textureSample(t, s, vec2f(0,0)).x == 0;
    `,
    uniform: false,
  },
  and_uniform_uniform: {
    code: `
      let x = uniform_cond && uniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: true,
  },
  and_uniform_nonuniform: {
    code: `
      let x = uniform_cond && nonuniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: false,
  },
  and_nonuniform_uniform: {
    code: `
      let x = nonuniform_cond && uniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: false,
  },
  and_nonuniform_nonuniform: {
    code: `
      let x = nonuniform_cond && nonuniform_cond;
      if x {
        let tmp = textureSample(t, s, vec2f(0,0));
      }
    `,
    uniform: false,
  },
  and_uniform_first_nonuniform: {
    code: `
      let x = textureSample(t, s, vec2f(0,0)).x == 0 && nonuniform_cond;
    `,
    uniform: true,
  },
  and_uniform_second_nonuniform: {
    code: `
      let x = nonuniform_cond && textureSample(t, s, vec2f(0,0)).x == 0;
    `,
    uniform: false,
  },
};

g.test('short_circuit_expressions')
  .desc(`Test uniformity of expressions`)
  .params(u => u.combine('case', keysOf(kShortCircuitExpressionCases)))
  .fn(t => {
    const testcase = kShortCircuitExpressionCases[t.params.case];
    const code = `
@group(1) @binding(0)
var t : texture_2d<f32>;
@group(1) @binding(1)
var s : sampler;

const uniform_cond = true;
var<private> nonuniform_cond = false;

@fragment
fn main() {
  ${testcase.code}
}
`;

    const res = testcase.uniform;
    if (!res) {
      t.expectCompileResult(true, `diagnostic(off, derivative_uniformity);\n` + code);
    }
    t.expectCompileResult(res, code);
  });

const kExpressionCases = {
  literal: {
    code: `1u`,
    uniform: true,
  },
  uniform: {
    code: `uniform_val`,
    uniform: true,
  },
  nonuniform: {
    code: `nonuniform_val`,
    uniform: false,
  },
  uniform_index: {
    code: `uniform_value[uniform_val]`,
    uniform: true,
  },
  nonuniform_index1: {
    code: `uniform_value[nonuniform_val]`,
    uniform: false,
  },
  nonuniform_index2: {
    code: `nonuniform_value[uniform_val]`,
    uniform: false,
  },
  uniform_struct: {
    code: `uniform_struct.x`,
    uniform: true,
  },
  nonuniform_struct: {
    code: `nonuniform_struct.x`,
    uniform: false,
  },
};

const kBinOps = {
  plus: {
    code: '+',
    test: '> 0',
  },
  minus: {
    code: '-',
    test: '> 0',
  },
  times: {
    code: '*',
    test: '> 0',
  },
  div: {
    code: '/',
    test: '> 0',
  },
  rem: {
    code: '%',
    test: '> 0',
  },
  and: {
    code: '&',
    test: '> 0',
  },
  or: {
    code: '|',
    test: '> 0',
  },
  xor: {
    code: '^',
    test: '> 0',
  },
  shl: {
    code: '<<',
    test: '> 0',
  },
  shr: {
    code: '>>',
    test: '> 0',
  },
  less: {
    code: '<',
    test: '',
  },
  lessequal: {
    code: '<=',
    test: '',
  },
  greater: {
    code: '>',
    test: '',
  },
  greaterequal: {
    code: '>=',
    test: '',
  },
  equal: {
    code: '==',
    test: '',
  },
  notequal: {
    code: '!=',
    test: '',
  },
};

g.test('binary_expressions')
  .desc(`Test uniformity of binary expressions`)
  .params(u =>
    u
      .combine('e1', keysOf(kExpressionCases))
      .combine('e2', keysOf(kExpressionCases))
      .combine('op', keysOf(kBinOps))
  )
  .fn(t => {
    const e1 = kExpressionCases[t.params.e1];
    const e2 = kExpressionCases[t.params.e2];
    const op = kBinOps[t.params.op];
    const code = `
@group(0) @binding(0)
var t : texture_2d<f32>;
@group(0) @binding(1)
var s : sampler;

struct S {
  x : u32
}

const uniform_struct = S(1);
var<private> nonuniform_struct = S(1);

const uniform_value : array<u32, 2> = array(1,1);
var<private> nonuniform_value : array<u32, 2> = array(1,1);

const uniform_val : u32 = 1;
var<private> nonuniform_val : u32 = 1;

@fragment
fn main() {
  let tmp = ${e1.code} ${op.code} ${e2.code};
  if tmp ${op.test} {
    let res = textureSample(t, s, vec2f(0,0));
  }
}
`;

    const res = e1.uniform && e2.uniform;
    if (!res) {
      t.expectCompileResult(true, `diagnostic(off, derivative_uniformity);\n` + code);
    }
    t.expectCompileResult(res, code);
  });

g.test('unary_expressions')
  .desc(`Test uniformity of uniary expressions`)
  .params(u =>
    u
      .combine('e', keysOf(kExpressionCases))
      .combine('op', ['!b_tmp', '~i_tmp > 0', '-i32(i_tmp) > 0'] as const)
  )
  .fn(t => {
    const e = kExpressionCases[t.params.e];
    const code = `
@group(0) @binding(0)
var t : texture_2d<f32>;
@group(0) @binding(1)
var s : sampler;

struct S {
  x : i32
}

const uniform_struct = S(1);
var<private> nonuniform_struct = S(1);

const uniform_value : array<i32, 2> = array(1,1);
var<private> nonuniform_value : array<i32, 2> = array(1,1);

const uniform_val : i32 = 1;
var<private> nonuniform_val : i32 = 1;

@fragment
fn main() {
  let i_tmp = ${e.code};
  let b_tmp = bool(i_tmp);
  let tmp = ${t.params.op};
  if tmp {
    let res = textureSample(t, s, vec2f(0,0));
  }
}
`;

    const res = e.uniform;
    if (!res) {
      t.expectCompileResult(true, `diagnostic(off, derivative_uniformity);\n` + code);
    }
    t.expectCompileResult(res, code);
  });