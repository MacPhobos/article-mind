# Article Mind UI - Frontend Development Guide

## Project Overview

SvelteKit-based frontend for the Article Mind knowledge management system. This application provides a modern, reactive user interface for managing and exploring article collections with AI-powered insights.

## Technology Stack

- **Framework:** SvelteKit 2.x with Svelte 5 (Runes API)
- **Language:** TypeScript 5.x (strict mode)
- **Build Tool:** Vite 6.x
- **Package Manager:** npm
- **Node Version:** 22.13.1 (managed by ASDF)

## Development Commands

### Standard Workflow

```bash
make install    # Install dependencies
make dev        # Start dev server (http://localhost:5173)
make build      # Build for production
make preview    # Preview production build
make test       # Run unit tests
make lint       # Check code quality
make format     # Format code
make check      # Run all quality checks (lint + format + test)
```

### API Type Generation

When the backend API changes:

```bash
make gen-api
# or
npm run gen:api
```

This fetches the OpenAPI spec from http://localhost:8000/openapi.json and generates TypeScript types in `src/lib/api/generated.ts`.

## Project Structure

```
src/
├── lib/
│   ├── api/              # API client and generated types
│   │   ├── client.ts     # Base API client
│   │   └── generated.ts  # Generated from OpenAPI (DO NOT EDIT MANUALLY)
│   └── components/       # Reusable Svelte components
├── routes/
│   ├── +layout.svelte    # Root layout with header
│   └── +page.svelte      # Home page
└── app.html              # HTML template
```

## API Contract Integration

### The Golden Rule

**NEVER manually edit `src/lib/api/generated.ts`**

This file is generated from the backend's OpenAPI specification. Manual edits will be overwritten.

### API Contract Workflow

1. Backend updates API contract in `article-mind-service/docs/api-contract.md`
2. Backend implements changes with Pydantic models
3. Backend tests pass
4. Frontend runs `make gen-api` to regenerate types
5. Frontend updates code using new types
6. Frontend tests pass

### Using the API Client

```typescript
import { apiClient } from '$lib/api/client';
import type { HealthResponse } from '$lib/api/generated';

const health = await apiClient.get<HealthResponse>('/health');
console.log(health.status); // Type-safe!
```

### Error Handling

The API client throws errors for failed requests. Always wrap API calls in try-catch blocks:

```typescript
try {
	const data = await apiClient.get('/api/v1/articles');
	// Handle success
} catch (error) {
	console.error('Failed to fetch articles:', error);
	// Handle error
}
```

## Svelte 5 Runes API

This project uses Svelte 5 with the Runes API for state management.

### Reactive State

Use `$state` to create reactive variables:

```svelte
<script lang="ts">
	let count = $state(0); // Reactive state

	function increment() {
		count++; // Automatically triggers re-render
	}
</script>

<button onclick={increment}>
	Count: {count}
</button>
```

### Derived State

Use `$derived` for computed values:

```svelte
<script lang="ts">
	let count = $state(0);
	let doubled = $derived(count * 2); // Auto-updates when count changes
</script>

<p>Count: {count}, Doubled: {doubled}</p>
```

### Effects

Use `$effect` for side effects:

```svelte
<script lang="ts">
	let search = $state('');

	$effect(() => {
		console.log('Search changed:', search);
		// Fetch results, etc.
	});
</script>
```

### Props

Use `$props()` to receive component props:

```svelte
<script lang="ts">
	interface Props {
		title: string;
		count?: number;
	}

	let { title, count = 0 }: Props = $props();
</script>

<h1>{title}</h1><p>Count: {count}</p>
```

## Code Quality Standards

### ESLint Rules

- Enforce TypeScript strict mode
- Svelte-specific linting (accessibility, best practices)
- No unused variables or imports
- Consistent code style

### Prettier Configuration

- Tab width: 1 tab (not spaces)
- Single quotes
- Trailing commas: none
- Print width: 100 characters
- Svelte formatting via prettier-plugin-svelte

### Running Quality Checks

```bash
make lint          # ESLint
make format-check  # Prettier dry-run
make format        # Auto-format all files
make check         # Run all checks (lint + format + test)
```

## Testing Strategy

### Unit Tests (Vitest)

- Test business logic in `$lib` modules
- Test component behavior with @testing-library/svelte
- Location: `tests/unit/`

```bash
make test         # Run once (CI mode)
make test-watch   # Watch mode
```

### Writing Tests

```typescript
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/svelte';
import MyComponent from '$lib/components/MyComponent.svelte';

describe('MyComponent', () => {
	it('renders correctly', () => {
		const { getByText } = render(MyComponent, { props: { title: 'Test' } });
		expect(getByText('Test')).toBeTruthy();
	});
});
```

## Environment Variables

Configure via `.env` file (never commit `.env`, use `.env.example`):

```env
VITE_API_BASE_URL=http://localhost:8000
```

Access in code:

```typescript
const apiUrl = import.meta.env.VITE_API_BASE_URL;
```

## Common Pitfalls

### 1. Editing Generated Types

❌ **NEVER DO THIS:**

```typescript
// In src/lib/api/generated.ts
export interface HealthResponse {
	status: string;
	custom_field: string; // WRONG - will be overwritten
}
```

✅ **DO THIS INSTEAD:**

- Update backend Pydantic models
- Run `make gen-api`

### 2. API Base URL Hardcoding

❌ **WRONG:**

```typescript
fetch('http://localhost:8000/api/v1/health');
```

✅ **CORRECT:**

```typescript
apiClient.get('/api/v1/health');
```

### 3. Forgetting to Regenerate Types

After backend API changes, ALWAYS run:

```bash
make gen-api
```

### 4. Not Using Type Guards

When working with API responses, use type guards:

```typescript
function isHealthResponse(data: unknown): data is HealthResponse {
	return typeof data === 'object' && data !== null && 'status' in data;
}
```

### 5. Forgetting Svelte 5 Runes Syntax

❌ **OLD (Svelte 4):**

```svelte
<script lang="ts">
	export let count = 0; // Svelte 4 prop syntax
</script>
```

✅ **NEW (Svelte 5):**

```svelte
<script lang="ts">
	let { count = 0 } = $props(); // Svelte 5 Runes syntax
</script>
```

### 6. Watch Mode in CI Environments

❌ **WRONG:**

```bash
npm test  # May trigger watch mode
```

✅ **CORRECT:**

```bash
make test  # Uses vitest run (no watch)
```

## Development Workflow

1. **Start dev server:** `make dev`
2. **Make changes** to Svelte components or TypeScript
3. **Hot reload** happens automatically
4. **Run tests:** `make test` (in watch mode: `make test-watch`)
5. **Check quality:** `make lint && make format-check`
6. **Format code:** `make format`
7. **Build:** `make build`

## ASDF Version Management

Node.js version is pinned in `.tool-versions`:

```
nodejs 22.13.1
```

To install:

```bash
asdf install
```

## Component Development Guidelines

### File Naming

- Components: PascalCase (e.g., `ArticleCard.svelte`)
- Routes: kebab-case with + prefix (e.g., `+page.svelte`, `+layout.svelte`)
- Utilities: camelCase (e.g., `formatDate.ts`)

### Component Structure

```svelte
<script lang="ts">
	// 1. Imports
	import { apiClient } from '$lib/api/client';

	// 2. Props
	interface Props {
		title: string;
	}
	let { title }: Props = $props();

	// 3. State
	let count = $state(0);

	// 4. Derived state
	let doubled = $derived(count * 2);

	// 5. Functions
	function increment() {
		count++;
	}

	// 6. Effects
	$effect(() => {
		console.log('Count changed:', count);
	});
</script>

<!-- 7. Template -->
<div>
	<h1>{title}</h1>
	<button onclick={increment}>Count: {count}</button>
</div>

<!-- 8. Styles -->
<style>
	div {
		padding: 1rem;
	}
</style>
```

### Accessibility

- Use semantic HTML elements
- Add ARIA labels where appropriate
- Ensure keyboard navigation works
- ESLint will warn about common accessibility issues

## Guard Rails

- ✅ **DO:** Use the API client for all backend requests
- ✅ **DO:** Regenerate types after backend changes
- ✅ **DO:** Write tests for new components
- ✅ **DO:** Run `make check` before committing
- ✅ **DO:** Use Svelte 5 Runes API for all state management
- ❌ **DON'T:** Edit generated type files manually
- ❌ **DON'T:** Hardcode API URLs
- ❌ **DON'T:** Commit `.env` files
- ❌ **DON'T:** Skip type checking (`// @ts-ignore` should be rare)
- ❌ **DON'T:** Use Svelte 4 syntax (no `export let` for props)

## Troubleshooting

### Dev Server Won't Start

**Issue:** Port 5173 already in use

**Solution:**

```bash
lsof -ti:5173 | xargs kill -9
make dev
```

### TypeScript Errors

**Issue:** `.svelte-kit/tsconfig.json` missing

**Solution:**

```bash
make dev  # Generates .svelte-kit directory
```

### API Type Generation Fails

**Issue:** `make gen-api` errors

**Solution:**

- Ensure backend is running on http://localhost:8000
- Verify `/openapi.json` endpoint is accessible
- Check `openapi-typescript` is installed

### Hot Reload Not Working

**Issue:** Changes don't reflect in browser

**Solution:**

```bash
# Restart dev server
make dev
# Clear browser cache
# Check Vite config has proper HMR settings
```

### ESLint and Prettier Conflicts

**Issue:** Prettier formats code, ESLint complains

**Solution:**

- Ensure all dependencies are installed
- Run `make format` then `make lint`

## Resources

- [SvelteKit Docs](https://kit.svelte.dev/docs)
- [Svelte 5 Runes](https://svelte.dev/docs/svelte/what-are-runes)
- [Vite Documentation](https://vitejs.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vitest Documentation](https://vitest.dev/)
- [Testing Library - Svelte](https://testing-library.com/docs/svelte-testing-library/intro)

## Next Steps

After scaffolding is complete:

1. Implement health check display (Plan 05)
2. Generate API types from backend: `make gen-api`
3. Create reusable components
4. Add routing for article management
5. Implement state management for complex features
