---
description: Refactor, clean up, and optimize code for readability, maintainability, and performance
model: claude-sonnet-4-5
---

Analyze and improve the following code for readability, maintainability, and performance.

## Code to Improve

$ARGUMENTS

## Improvement Strategy

### 1. **Code Quality Assessment**

Evaluate the code for:
- **Readability**: Can someone new understand this quickly?
- **Maintainability**: How easy is it to modify?
- **Performance**: Are there obvious bottlenecks?
- **Type Safety**: Are types properly annotated?

---

## Part A: Code Cleanup & Refactoring

### 2. **Code Smells to Fix**

**Naming**
-  Descriptive variable/function names
-  Consistent naming conventions (camelCase, PascalCase)
-  Avoid abbreviations unless obvious
-  Boolean names start with is/has/can

**Functions**
-  Single responsibility per function
-  Keep functions small (<50 lines)
-  Reduce parameters (max 3-4)
-  Extract complex logic
-  Avoid side effects where possible

**DRY (Don't Repeat Yourself)**
-  Extract repeated code to utilities
-  Create reusable components
-  Use TypeScript generics for type reuse
-  Centralize constants/configuration

**Complexity**
-  Reduce nested if statements
-  Replace complex conditions with functions
-  Use early returns
-  Simplify boolean logic

**TypeScript**
-  Remove `any` types
-  Add proper type annotations
-  Use interfaces for object shapes
-  Leverage utility types (Pick, Omit, Partial)

### 3. **Modern Patterns to Apply**

**JavaScript/TypeScript**
```typescript
// Use optional chaining
const value = obj?.prop?.nested

// Use nullish coalescing
const result = value ?? defaultValue

// Use destructuring
const { name, email } = user

// Use template literals
const message = `Hello, ${name}!`

// Use array methods
const filtered = arr.filter(x => x.active)
```

**React**
```typescript
// Extract custom hooks
const useUserData = () => {
  // logic here
}

// Use proper TypeScript types
interface Props {
  user: User
  onUpdate: (user: User) => void
}

// Avoid prop drilling with composition
<Provider value={data}>
  <Component />
</Provider>
```

### 4. **Refactoring Techniques**

**Extract Function**
```typescript
// Before
const process = () => {
  // 50 lines of code
}

// After
const validate = () => { /* ... */ }
const transform = () => { /* ... */ }
const save = () => { /* ... */ }

const process = () => {
  validate()
  const data = transform()
  save(data)
}
```

**Replace Conditional with Polymorphism**
```typescript
// Before
if (type === 'A') return processA()
if (type === 'B') return processB()

// After
const processors = {
  A: processA,
  B: processB
}
return processors[type]()
```

**Introduce Parameter Object**
```typescript
// Before
function create(name, email, age, address)

// After
interface UserData {
  name: string
  email: string
  age: number
  address: string
}
function create(userData: UserData)
```

### 5. **Common Cleanup Tasks**

**Remove Dead Code**
- Unused imports
- Unreachable code
- Commented out code
- Unused variables

**Improve Error Handling**
```typescript
// Before
try { doSomething() } catch (e) { console.log(e) }

// After
try {
  doSomething()
} catch (error) {
  if (error instanceof ValidationError) {
    // Handle validation
  } else {
    logger.error('Unexpected error', { error })
    throw error
  }
}
```

**Consistent Formatting**
- Proper indentation
- Consistent quotes
- Line length (<100 characters)
- Organized imports

---

## Part B: Performance Optimization

### 6. **Profiling First**
- Identify actual bottlenecks
- Don't optimize prematurely
- Measure before and after
- Focus on high-impact areas

### 7. **Performance Optimization Areas**

**React/Next.js**
- Memoization (React.memo, useMemo, useCallback)
- Code splitting and lazy loading
- Image optimization (next/image)
- Font optimization (next/font)
- Remove unnecessary re-renders
- Virtual scrolling for long lists

**Database Queries**
- Add indexes for frequently queried fields
- Batch queries (reduce N+1 problems)
- Use select to limit fields
- Implement pagination
- Cache frequent queries
- Use database views for complex joins

**API Calls**
- Implement caching (SWR, React Query)
- Debounce/throttle requests
- Parallel requests where possible
- Request deduplication
- Optimistic updates

**Bundle Size**
- Tree-shaking unused code
- Dynamic imports for large libraries
- Replace heavy dependencies
- Code splitting by route
- Lazy load below-the-fold content

**Memory**
- Fix memory leaks (cleanup useEffect)
- Avoid unnecessary object creation
- Use const for non-changing values
- Clear intervals/timeouts
- Remove event listeners

### 8. **Optimization Checklist**

**JavaScript/TypeScript**
-  Use const/let instead of var
-  Avoid nested loops where possible
-  Use Map/Set for lookups
-  Minimize DOM manipulations
-  Debounce/throttle expensive operations

**React**
-  Memo components that render often
-  Move static values outside components
-  Use keys properly in lists
-  Avoid inline functions in render
-  Lazy load routes and components

**Next.js**
-  Use Server Components where possible
-  Implement ISR for dynamic content
-  Optimize images with next/image
-  Prefetch critical routes
-  Use Suspense for streaming

**Database**
-  Add indexes on foreign keys
-  Use prepared statements
-  Batch inserts/updates
-  Implement connection pooling
-  Cache expensive queries

**Network**
-  Compress responses (gzip/brotli)
-  Use CDN for static assets
-  Implement HTTP/2
-  Set proper cache headers
-  Minimize payload size

### 9. **Common Optimizations**

**Replace inefficient array methods**
```typescript
// Before: Multiple iterations
const result = arr
  .filter(x => x > 0)
  .map(x => x * 2)
  .reduce((sum, x) => sum + x, 0)

// After: Single iteration
const result = arr.reduce((sum, x) => {
  return x > 0 ? sum + (x * 2) : sum
}, 0)
```

**Memoize expensive calculations**
```typescript
const expensiveValue = useMemo(() => {
  return complexCalculation(props.data)
}, [props.data])
```

**Virtual scrolling for long lists**
```typescript
import { useVirtual } from 'react-virtual'
// Only render visible items
```

### 10. **Measurement Tools**

**Frontend**
- Chrome DevTools Performance tab
- Lighthouse CI
- React DevTools Profiler
- Bundle Analyzer (next/bundle-analyzer)

**Backend**
- Node.js profiler
- Database query analyzer
- APM tools (DataDog, New Relic)
- Load testing (k6, Artillery)

---

## Output Format

1. **Issues Found** - List of code smells, problems, and bottlenecks
2. **Improved Code** - Refactored and optimized version
3. **Explanations** - What changed and why
4. **Before/After Comparison** - Side-by-side if helpful
5. **Performance Impact** - Expected improvement metrics
6. **Trade-offs** - Any complexity added
7. **Further Improvements** - Optional enhancements

Focus on practical improvements that provide real value. Balance clean code with pragmatism and don't sacrifice readability for micro-optimizations.
