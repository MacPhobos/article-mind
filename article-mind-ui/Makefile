.PHONY: help install dev build preview test test-watch lint format format-check clean check gen-api

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_:-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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

check: ## Run all quality checks (lint + format-check + test)
	npm run lint && npm run format:check && npm run test

gen-api: ## Generate API types from OpenAPI spec
	npm run gen:api

clean: ## Remove build artifacts and node_modules
	rm -rf .svelte-kit build node_modules
