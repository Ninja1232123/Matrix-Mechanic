/**
 * Example JavaScript file with various errors for testing the AI Debug Companion.
 *
 * Run with: node javascript_errors.js
 */


function referenceErrorExample() {
    // ReferenceError: variable not defined
    console.log(undefinedVariable);
}


function typeErrorExample() {
    // TypeError: cannot read property of undefined
    const obj = undefined;
    console.log(obj.property);
}


function typeErrorFunctionExample() {
    // TypeError: not a function
    const notAFunction = "hello";
    notAFunction();
}


function rangeErrorExample() {
    // RangeError: Maximum call stack size exceeded
    function recursiveFunction() {
        recursiveFunction();
    }
    recursiveFunction();
}


function syntaxErrorExample() {
    // This would cause a SyntaxError if uncommented:
    // const broken = {
    //     key: "value"
    //     missing: "comma"
    // };
}


function moduleNotFoundExample() {
    // Error: Cannot find module
    require('non-existent-module');
}


// Run one example at a time
console.log("Running JavaScript error examples...");
console.log("Uncomment one function at a time to test different errors.\n");

// Uncomment to test:
// referenceErrorExample();
// typeErrorExample();
// typeErrorFunctionExample();
// rangeErrorExample();
moduleNotFoundExample();
