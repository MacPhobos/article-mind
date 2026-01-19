# Plan 01: SvelteKit UI Scaffolding

**Plan ID:** 01-scaffold-ui
**Created:** 2026-01-18
**Dependencies:** None
**Estimated Time:** 2-4 hours

---

## 1. Overview

### Purpose
Scaffold the article-mind-ui frontend application using SvelteKit, TypeScript, and Vite with a complete development toolchain including linting, formatting, and testing infrastructure.

### Scope
- Initialize SvelteKit project with TypeScript
- Configure build tooling (Vite)
- Set up code quality tools (ESLint, Prettier)
- Create Makefile for common commands
- Configure ASDF for version management
- Establish project-specific CLAUDE.md

### Dependencies
None - this is a foundational plan.

### Outputs
- Working SvelteKit application skeleton
- Development server running on http://localhost:5173
- Complete toolchain ready for feature development

---

## 2. Technology Stack

### Core Framework
- **SvelteKit:** 2.x (latest stable)
- **Svelte:** 5.x (latest with Runes API)
- **TypeScript:** 5.x
- **Node.js:** 22.x LTS (via ASDF)

### Build Tooling
- **Vite:** 6.x (bundler, dev server, HMR)
- **@sveltejs/adapter-auto:** Deployment adapter
- **@sveltejs/vite-plugin-svelte:** Svelte integration

### Code Quality
- **ESLint:** 9.x with TypeScript parser
- **eslint-plugin-svelte:** Svelte-specific linting rules
- **Prettier:** 3.x with prettier-plugin-svelte
- **prettier-plugin-svelte:** Svelte formatting

### Testing (Initial Setup)
- **Vitest:** Unit testing framework
- **@testing-library/svelte:** Component testing utilities

### Package Manager
- **npm:** (comes with Node.js via ASDF)

---

## 3. Directory Structure

```
article-mind/
└── article-mind-ui/
    ├── .tool-versions          # ASDF version pinning
    ├── Makefile                # Common commands
    ├── CLAUDE.md               # Frontend-specific instructions
    ├── package.json            # Dependencies and scripts
    ├── tsconfig.json           # TypeScript configuration
    ├── vite.config.ts          # Vite build configuration
    ├── svelte.config.js        # SvelteKit configuration
    ├── .eslintrc.cjs           # ESLint configuration
    ├── .prettierrc             # Prettier configuration
    ├── .prettierignore         # Prettier ignore patterns
    ├── src/
    │   ├── app.html            # HTML template
    │   ├── app.css             # Global styles
    │   ├── lib/
    │   │   ├── api/            # API client code
    │   │   │   └── client.ts   # Base API client
    │   │   └── components/     # Reusable components
    │   └── routes/
    │       ├── +layout.svelte  # Root layout
    │       └── +page.svelte    # Home page
    ├── static/
    │   └── favicon.png         # Static assets
    └── tests/
        └── unit/               # Unit tests
```

---

## 4. Implementation Steps

### Step 1: Initialize Project with SvelteKit

```bash
cd /export/workspace/article-mind
npm create svelte@latest article-mind-ui
```

**Interactive Prompts:**
- Template: Skeleton project
- TypeScript: Yes, using TypeScript syntax
- ESLint: Yes
- Prettier: Yes
- Playwright: No (defer browser testing)
- Vitest: Yes

### Step 2: Configure ASDF Version Management

Create `.tool-versions`:

```bash
cd article-mind-ui
cat > .tool-versions << 'EOF'
nodejs 22.13.1
EOF
```

Install Node.js via ASDF:

```bash
asdf install nodejs 22.13.1
asdf reshim nodejs
```

### Step 3: Install Dependencies

```bash
npm install
```

### Step 4: Create Makefile

Create `Makefile`:

```makefile
.PHONY: help install dev build test lint format clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	npm install

dev: ## Start development server
	npm run dev

build: ## Build for production
	npm run build

preview: ## Preview production build
	npm run preview

test: ## Run unit tests
	npm run test

test-watch: ## Run tests in watch mode
	npm run test:watch

lint: ## Run ESLint
	npm run lint

format: ## Format code with Prettier
	npm run format

format-check: ## Check code formatting
	npm run format:check

clean: ## Remove build artifacts and node_modules
	rm -rf .svelte-kit build node_modules
```

Make executable:

```bash
chmod +x Makefile
```

### Step 5: Configure package.json Scripts

Update `package.json` scripts section:

```json
{
  "scripts": {
    "dev": "vite dev --host",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint .",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "gen:api": "openapi-typescript http://localhost:8000/openapi.json -o src/lib/api/generated.ts"
  }
}
```

### Step 6: Install Additional Dependencies

```bash
npm install -D openapi-typescript
```

### Step 7: Create API Client Structure

```bash
mkdir -p src/lib/api
```

Create `src/lib/api/client.ts`:

```typescript
/**
 * Base API client for article-mind-service
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface ApiError {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const error: ApiError = await response.json();
        throw new Error(error.error.message || 'API request failed');
      }

      return response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.fetch<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data: unknown): Promise<T> {
    return this.fetch<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async patch<T>(endpoint: string, data: unknown): Promise<T> {
    return this.fetch<T>(endpoint, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.fetch<T>(endpoint, { method: 'DELETE' });
  }
}

export const apiClient = new ApiClient();
```

### Step 8: Configure Environment Variables

Create `.env.example`:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
```

Create `.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Add to `.gitignore`:

```
.env
.env.local
```

### Step 9: Update TypeScript Configuration

Update `tsconfig.json`:

```json
{
  "extends": "./.svelte-kit/tsconfig.json",
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "sourceMap": true,
    "strict": true,
    "moduleResolution": "bundler",
    "paths": {
      "$lib": ["./src/lib"],
      "$lib/*": ["./src/lib/*"]
    }
  }
}
```

### Step 10: Configure Vite

Update `vite.config.ts`:

```typescript
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [sveltekit()],
  test: {
    include: ['tests/**/*.{test,spec}.{js,ts}']
  },
  server: {
    port: 5173,
    host: true
  }
});
```

### Step 11: Create CLAUDE.md

Create `CLAUDE.md`:

```markdown
# Article Mind UI - Frontend Development Guide

## Project Overview

SvelteKit-based frontend for the Article Mind knowledge management system.

## Technology Stack

- **Framework:** SvelteKit 2.x with Svelte 5 (Runes API)
- **Language:** TypeScript 5.x
- **Build Tool:** Vite 6.x
- **Package Manager:** npm
- **Node Version:** 22.13.1 (managed by ASDF)

## Development Commands

### Standard Workflow

```bash
make install    # Install dependencies
make dev        # Start dev server (http://localhost:5173)
make build      # Build for production
make test       # Run unit tests
make lint       # Check code quality
make format     # Format code
```

### API Type Generation

When the backend API changes:

```bash
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
4. Frontend runs `npm run gen:api` to regenerate types
5. Frontend updates code using new types
6. Frontend tests pass

### Using the API Client

```typescript
import { apiClient } from '$lib/api/client';
import type { HealthResponse } from '$lib/api/generated';

const health = await apiClient.get<HealthResponse>('/health');
console.log(health.status); // Type-safe!
```

## Code Quality Standards

### ESLint Rules

- Enforce TypeScript strict mode
- Svelte-specific linting (accessibility, best practices)
- No unused variables or imports
- Consistent code style

### Prettier Configuration

- Tab width: 2 spaces
- Single quotes
- Trailing commas: ES5
- Svelte formatting via prettier-plugin-svelte

### Running Quality Checks

```bash
make lint          # ESLint
make format-check  # Prettier dry-run
make format        # Auto-format all files
```

## Testing Strategy

### Unit Tests (Vitest)

- Test business logic in `$lib` modules
- Test component behavior with @testing-library/svelte
- Location: `tests/unit/`

```bash
make test         # Run once
make test-watch   # Watch mode
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
  custom_field: string;  // WRONG - will be overwritten
}
```

✅ **DO THIS INSTEAD:**
- Update backend Pydantic models
- Run `npm run gen:api`

### 2. API Base URL Hardcoding

❌ **WRONG:**
```typescript
fetch('http://localhost:8000/api/v1/health')
```

✅ **CORRECT:**
```typescript
apiClient.get('/api/v1/health')
```

### 3. Forgetting to Regenerate Types

After backend API changes, ALWAYS run:

```bash
npm run gen:api
```

### 4. Not Using Type Guards

When working with API responses, use type guards:

```typescript
function isHealthResponse(data: unknown): data is HealthResponse {
  return typeof data === 'object' && data !== null && 'status' in data;
}
```

## Development Workflow

1. **Start dev server:** `make dev`
2. **Make changes** to Svelte components or TypeScript
3. **Hot reload** happens automatically
4. **Run tests:** `make test` (in watch mode: `make test-watch`)
5. **Check quality:** `make lint && make format-check`
6. **Format code:** `make format`
7. **Build:** `make build`

## Svelte 5 Runes API

This project uses Svelte 5 with the Runes API for state management:

```svelte
<script lang="ts">
  let count = $state(0);  // Reactive state
  let doubled = $derived(count * 2);  // Derived state

  function increment() {
    count++;
  }
</script>

<button onclick={increment}>
  Count: {count}, Doubled: {doubled}
</button>
```

## ASDF Version Management

Node.js version is pinned in `.tool-versions`:

```
nodejs 22.13.1
```

To install:

```bash
asdf install
```

## Guard Rails

- ✅ **DO:** Use the API client for all backend requests
- ✅ **DO:** Regenerate types after backend changes
- ✅ **DO:** Write tests for new components
- ✅ **DO:** Run `make lint` before committing
- ❌ **DON'T:** Edit generated type files manually
- ❌ **DON'T:** Hardcode API URLs
- ❌ **DON'T:** Commit `.env` files
- ❌ **DON'T:** Skip type checking (`// @ts-ignore` should be rare)

## Resources

- [SvelteKit Docs](https://kit.svelte.dev/docs)
- [Svelte 5 Runes](https://svelte.dev/docs/svelte/what-are-runes)
- [Vite Documentation](https://vitejs.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
```

### Step 12: Create Initial Root Layout

Create `src/routes/+layout.svelte`:

```svelte
<script lang="ts">
  // Root layout - wraps all pages
</script>

<main>
  <slot />
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: system-ui, -apple-system, sans-serif;
  }

  main {
    min-height: 100vh;
  }
</style>
```

### Step 13: Create Initial Home Page

Update `src/routes/+page.svelte`:

```svelte
<script lang="ts">
  let appName = $state('Article Mind');
</script>

<div class="container">
  <h1>Welcome to {appName}</h1>
  <p>SvelteKit frontend is running!</p>
</div>

<style>
  .container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    text-align: center;
  }

  h1 {
    color: #333;
    margin-bottom: 1rem;
  }
</style>
```

---

## 5. CLAUDE.md Requirements

The project-specific CLAUDE.md should document:

### Essential Information
- ✅ Technology stack (SvelteKit, TypeScript, Vite)
- ✅ Node.js version (22.13.1 via ASDF)
- ✅ Package manager (npm)
- ✅ Development commands (Makefile targets)

### API Contract Integration
- ✅ API type generation workflow
- ✅ Generated types file location and DO NOT EDIT warning
- ✅ API client usage examples
- ✅ Backend → Frontend sync process

### Code Quality Standards
- ✅ ESLint and Prettier configuration
- ✅ Testing strategy (Vitest)
- ✅ Environment variable management

### Guard Rails
- ✅ Never edit generated types manually
- ✅ Always use API client for backend requests
- ✅ Never hardcode API URLs
- ✅ Run type generation after backend changes
- ✅ Never commit `.env` files

### Project Structure
- ✅ Directory layout explanation
- ✅ Component organization patterns
- ✅ API client architecture

---

## 6. Verification Steps

### 1. Verify Installation

```bash
cd article-mind-ui
make install
```

**Expected:** No errors, `node_modules/` created.

### 2. Verify Dev Server

```bash
make dev
```

**Expected:**
- Server starts on http://localhost:5173
- Browser opens showing "Welcome to Article Mind"
- Hot reload works (edit `+page.svelte`, see changes instantly)

### 3. Verify Build

```bash
make build
```

**Expected:**
- Build succeeds in `.svelte-kit/output`
- No TypeScript errors
- No build warnings

### 4. Verify Linting

```bash
make lint
```

**Expected:** No linting errors.

### 5. Verify Formatting

```bash
make format-check
```

**Expected:** All files formatted correctly.

### 6. Verify Tests

```bash
make test
```

**Expected:** Initial test suite passes (even if minimal).

### 7. Verify ASDF Version

```bash
node --version
```

**Expected:** `v22.13.1` (or exact version in `.tool-versions`).

### 8. Verify Makefile

```bash
make help
```

**Expected:** Help message with all available targets.

### 9. Verify API Client

Create a simple test to verify API client structure:

```typescript
import { apiClient } from '$lib/api/client';

// This should not throw type errors
const client = apiClient;
console.log(client);
```

### 10. Verify Environment Variables

```bash
cat .env
```

**Expected:** `VITE_API_BASE_URL=http://localhost:8000`

---

## 7. Common Pitfalls

### Issue 1: Node Version Mismatch

**Symptom:** `npm install` fails with version errors.

**Solution:**
```bash
asdf install nodejs 22.13.1
asdf reshim nodejs
node --version  # Verify
```

### Issue 2: Port 5173 Already in Use

**Symptom:** Dev server fails to start.

**Solution:**
```bash
# Kill existing process
lsof -ti:5173 | xargs kill -9

# Or use different port
vite dev --port 5174
```

### Issue 3: TypeScript Errors After SvelteKit Scaffold

**Symptom:** `.svelte-kit/tsconfig.json` missing.

**Solution:**
```bash
npm run dev  # Generates .svelte-kit directory
```

### Issue 4: Prettier and ESLint Conflicts

**Symptom:** Prettier formats code, ESLint complains.

**Solution:** Ensure `eslint-config-prettier` is installed (should be via SvelteKit template).

### Issue 5: ASDF Not Found

**Symptom:** `asdf: command not found`

**Solution:**
```bash
# Install ASDF first
git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.14.0
echo '. "$HOME/.asdf/asdf.sh"' >> ~/.bashrc
source ~/.bashrc

# Install Node.js plugin
asdf plugin add nodejs
```

### Issue 6: API Type Generation Fails

**Symptom:** `npm run gen:api` errors.

**Solution:**
- Ensure backend is running on http://localhost:8000
- Verify `/openapi.json` endpoint is accessible
- Check `openapi-typescript` is installed

### Issue 7: Hot Reload Not Working

**Symptom:** Changes don't reflect in browser.

**Solution:**
```bash
# Restart dev server
make dev
# Clear browser cache
# Check Vite config has proper HMR settings
```

---

## 8. Next Steps

After completing this plan:

1. **Proceed to Plan 02** (scaffold-service) to create backend
2. **Implement health check display** in Plan 05 (ui-health-display)
3. **Set up API contract** following API contract instructions
4. **Generate types** with `npm run gen:api` once backend is running

---

## 9. Success Criteria

- ✅ SvelteKit application runs on http://localhost:5173
- ✅ TypeScript compilation succeeds with no errors
- ✅ ESLint and Prettier pass with no violations
- ✅ Makefile provides all standard commands (install, dev, build, test, lint, format)
- ✅ ASDF pins Node.js version correctly
- ✅ API client structure exists and is type-safe
- ✅ Environment variables configured
- ✅ CLAUDE.md documents all workflows and guard rails
- ✅ Hot reload works for development
- ✅ Production build succeeds

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-18
